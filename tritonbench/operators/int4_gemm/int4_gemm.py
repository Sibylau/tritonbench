"""
Compute a bf16 (activation) x int4 (weight) gemm.
Inspired by [gpt-fast](https://github.com/meta-pytorch/gpt-fast)
ATen kernels from tinygemm
Triton implementation by @jlebar: https://gist.github.com/jlebar/3435b2c00deea53258887ce37231e5e2
"""

import argparse
import statistics

from typing import Any, List, Optional

import torch
import triton
import triton.language as tl

from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    BenchmarkOperatorMetrics,
    register_benchmark,
    register_metric,
)

from .kernel import _group_quantize_tensor, matmul, matmul_kernel, pack_2xint4


class Operator(BenchmarkOperator):
    DEFAULT_METRICS = ["tflops", "gbps", "latency", "best_config"]
    FWD_ONLY = True

    def __init__(
        self, tb_args: argparse.Namespace, extra_args: Optional[List[str]] = None
    ):
        super().__init__(tb_args, extra_args)
        # `Group size` and `inner K tiles` are defaults from gpt-fast.
        self.group_size = 32
        self.inner_k_tiles = 8

    def get_input_iter(self):
        def args(B, L, Dout, Din):
            x = torch.randn(B, L, Din, device=self.device, dtype=torch.bfloat16)
            w = torch.randint(-8, 7, (Din, Dout), device=self.device, dtype=torch.int32)
            return (x, w)

        # LLama-2 shapes w/ 8-way tensor parallelism.
        name_to_shapes_70b = {
            "attn.wqkv": (8192, 1280),
            "attn.w0": (1024, 8192),
            "ffn.w13": (8192, 7168),
            "ffn.w2": (3584, 8192),
        }
        for seq_len in (1, 4096):
            for bsz in (1, 4, 16, 64):
                for name, (k, n) in name_to_shapes_70b.items():
                    yield args(bsz, seq_len, n, k)

    def get_x_val(self, example_inputs) -> float:
        x, w = example_inputs
        B, m, k = x.size()
        _, n = w.size()
        return (B, m, n, k)

    def _matmul_from_packed(self, x_2d, w_int4_packed, K, N):
        w_int8 = w_int4_packed.to(torch.int8)

        # Extract low and high 4-bit values with sign extension for low bits
        w_lo = ((w_int8 << 4) >> 4).to(torch.bfloat16)  # Sign extend lower 4-bit
        w_hi = (w_int8 >> 4).to(torch.bfloat16)  # Upper 4-bit

        # Interleave them back to get full K dimension
        w_unpacked = torch.stack([w_lo, w_hi], dim=1).reshape(K, N)

        # Perform regular matrix multiplication
        return torch.matmul(x_2d, w_unpacked)

    @register_benchmark(baseline=True)
    def eager_int4_gemm(self, x, w):
        def compute_unpack_and_matmul():
            x_2d = x.reshape(-1, x.size(-1))
            K, N = w.shape
            w_int4_packed = pack_2xint4(w).T.contiguous().T
            return self._matmul_from_packed(x_2d, w_int4_packed, K, N)

        return compute_unpack_and_matmul

    @register_benchmark()
    def torch_compile_int4_gemm(self, x, w):
        return torch.compile(
            self.eager_int4_gemm(x, w), mode="max-autotune-no-cudagraphs"
        )

    @register_benchmark()
    def triton_int4_gemm(self, x, w):
        def run_kernel():
            x_2d = x.reshape(-1, x.size(-1))
            w_int4_packed = pack_2xint4(w).T.contiguous().T

            return matmul(x_2d, w_int4_packed)

        return run_kernel

    @register_benchmark()
    def preprocessed_eager_int4_gemm(self, x, w):
        x_2d = x.reshape(-1, x.size(-1))
        K, N = w.shape
        w_int4_packed = pack_2xint4(w).T.contiguous().T

        return lambda: self._matmul_from_packed(x_2d, w_int4_packed, K, N)

    @register_benchmark()
    def preprocessed_torch_compile_int4_gemm(self, x, w):
        return torch.compile(
            self.preprocessed_eager_int4_gemm(x, w), mode="max-autotune-no-cudagraphs"
        )

    @register_benchmark()
    def preprocessed_triton_int4_gemm(self, x, w):
        x_2d = x.reshape(-1, x.size(-1))
        w_int4_packed = pack_2xint4(w).T.contiguous().T

        return lambda: matmul(x_2d, w_int4_packed)

    @register_metric()
    def gbps(self, fn, example_inputs: Any, metrics: BenchmarkOperatorMetrics) -> float:
        def nbytes(t):
            return t.numel() * t.element_size()

        x, w = example_inputs
        w, scales_and_zeros = _group_quantize_tensor(
            w.to(torch.bfloat16), n_bit=4, q_group_size=self.group_size
        )
        c = fn()

        gb = (sum(nbytes(t) for t in (x, scales_and_zeros, c)) + nbytes(w) // 8) / 1e9
        return gb / metrics.latency * 1e3

    @register_metric()
    def flops(
        self, fn_name: str, example_inputs: Any, metrics: BenchmarkOperatorMetrics
    ) -> float:
        a, b = example_inputs
        B, m, k = a.size()
        m = B * m
        _, n = b.size()
        flops = 2 * m * n * k
        return flops

    def accuracy(self, fn, baseline_fn):
        output = fn()
        baseline_output = baseline_fn()
        rtol = self.tb_args.rtol if self.tb_args.rtol is not None else 1e-2
        atol = self.tb_args.atol if self.tb_args.atol is not None else 8.0

        try:
            torch.testing.assert_close(
                output,
                baseline_output,
                rtol=rtol,
                atol=atol,
            )
            return True
        except AssertionError:
            return False

    def plot(self):
        @triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=[
                    "B",
                    "m",
                    "n",
                    "k",
                ],  # argument names to use as an x-axis for the plot
                x_vals=self.output.x_vals,  # different possible values for `x_name`
                line_arg="provider",  # argument name whose value corresponds to a different line in the plot
                line_vals=[
                    "eager_int4_gemm",
                    "triton_int4_gemm",
                ],  # possible values for `line_arg``
                line_names=[
                    "eager_int4_gemm",
                    "triton_int4_gemm",
                ],  # label name for the lines
                styles=[("blue", "-"), ("green", "-")],
                ylabel="tflops",  # label name for the y-axis
                plot_name="int4-gemm-performance",  # name for the plot. Used also as a file name for saving the plot.
                args={},  # values for function arguments not in `x_names` and `y_name`
            )
        )
        def _plot(B, m, n, k, provider):
            tflops = self.output.get_y_vals((B, m, n, k), provider, "tflops")
            return tflops

        save_path = "/tmp/int4_gemm"

        _plot.run(show_plots=True, print_data=True, save_path=save_path)
