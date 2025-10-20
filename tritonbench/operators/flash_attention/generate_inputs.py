from itertools import chain
from typing import Generator, Tuple

import torch


def _get_standard_shapes(shape, num_inputs, dtype, device) -> Generator:
    BATCH, H, SEQ_LEN, SEQ_LEN_KV, D_HEAD = shape
    if SEQ_LEN:
        if num_inputs is None:
            yield (BATCH, H, SEQ_LEN, SEQ_LEN_KV, D_HEAD)
        else:
            for _i in range(num_inputs):
                yield (BATCH, H, SEQ_LEN, SEQ_LEN_KV, D_HEAD)
                SEQ_LEN *= 2
        return

    SEQ_LEN_LOG2 = 7
    for i in range(SEQ_LEN_LOG2, 14):
        N_CTX = 2**i
        # BATCH = 16384 // N_CTX
        # H = 2048 // D_HEAD
        yield (BATCH, H, N_CTX, N_CTX, D_HEAD)


def _generated_qkv_inputs(
    shape, dtype, device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    requires_grad = True

    if len(shape) == 5:
        BATCH, H, N_CTX, N_CTX_KV, D_HEAD = shape
    else:
        BATCH, H, N_CTX, D_HEAD = shape
        N_CTX_KV = N_CTX
    q = torch.randn(
        (BATCH, H, N_CTX, D_HEAD),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        (BATCH, H, N_CTX_KV, D_HEAD),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    v = torch.randn(
        (BATCH, H, N_CTX_KV, D_HEAD),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return (q, k, v)


def standard_inputs(shape, num_inputs, dtype, device) -> Generator:
    for shape in _get_standard_shapes(shape, num_inputs, dtype, device):  # noqa
        yield _generated_qkv_inputs(shape, dtype, device)


def additional_inputs(
    shape,
    num_inputs,
    dtype,
    device,
    add_production_shapes,
    name,
    shuffle_shapes,
) -> Generator:
    standard_shapes = _get_standard_shapes(shape, num_inputs, dtype, device)
    llama_shapes = [
        (4, 32, 19, 128),
        (4, 32, 1, 128),
        # currently we are only able to use the same shape for q, k, v but in
        # prod q shape is (4, 32, 1, 128) here
        (4, 32, 511, 128),
    ]
    shapes = chain(standard_shapes, llama_shapes)
    if add_production_shapes:
        from ...utils.fb.durin_data import productionDataLoader

        shapes = chain(
            shapes,
            productionDataLoader.get_shapes_from_frozen_durin(
                name, "attention", shuffle_shapes=shuffle_shapes
            ),
        )
    for shape in shapes:
        yield _generated_qkv_inputs(shape, dtype, device)


def ragged_inputs(dtype, device) -> Generator:
    additional_shapes = [
        (1024, 4, 1024, 128),
        (256, 4, 256, 128),
        (256, 4, 512, 128),
        (256, 4, 1024, 128),
        (256, 4, 2048, 128),
        (256, 4, 4096, 128),
        (256, 4, 8192, 128),
        (256, 4, 16384, 128),
    ]
    for shape in chain(additional_shapes):
        yield _generated_qkv_inputs(shape, dtype, device)


def sweep_inputs(dtype, device) -> Generator:
    D = 128
    batch_sizes = [2**i for i in range(6)]
    num_heads = [1, 4, 8, 16]
    seqlen = [512 * (2**i) for i in range(6)]
    for B in batch_sizes:
        for H in num_heads:
            for S in seqlen:
                yield _generated_qkv_inputs(
                    shape=(B, H, S, D), dtype=dtype, device=device
                )
