# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Source-side packing primitives for the next combined EP implementation.

Quantize each original token once, before top-k duplication. Backward restores
the sum of every routed input derivative with int64 tensor addressing.
"""
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _invert_assignments(order, inverse, count, BLOCK: tl.constexpr):
    positions = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    original = tl.load(order + positions, mask=positions < count, other=0).to(tl.int64)
    tl.store(inverse + original, positions, mask=positions < count)


@triton.jit
def _restore_input(gradient, inverse, output, rows, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
                   BH: tl.constexpr):
    tokens = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    columns = (tl.program_id(1) * BH + tl.arange(0, BH)).to(tl.int64)
    choices = tl.arange(0, K)
    positions = tl.load(inverse + tokens[:, None] * K + choices[None, :], mask=tokens[:, None] < rows,
                        other=0).to(tl.int64)
    values = tl.load(gradient + positions[:, :, None] * H + columns[None, None, :],
                     mask=(tokens[:, None, None] < rows) & (columns[None, None, :] < H),
                     other=0).to(tl.float32)
    result = tl.sum(values, axis=1)
    tl.store(output + tokens[:, None] * H + columns[None, :],
             result,
             mask=(tokens[:, None] < rows) & (columns[None, :] < H))


def sorted_assignments(experts, num_experts=256):
    if num_experts >= 2**24:
        raise ValueError('FP32 expert sorting requires IDs below 2**24')
    return torch.argsort(experts.reshape(-1).float(), stable=True)


def pack_quantized_source(tokens, order, top_k, scale):
    payload = torch_npu.npu_quantize(tokens,
                                     scales=scale,
                                     zero_points=None,
                                     dtype=torch_npu.hifloat8,
                                     axis=1,
                                     div_mode=True)
    return payload.index_select(0, order // top_k)


def inverse_assignments(order):
    output = torch.empty_like(order)
    _invert_assignments[(triton.cdiv(order.numel(), 512), )](order, output, order.numel(), BLOCK=512)
    return output


def restore_input_gradient(gradient, inverse, rows, top_k):
    if top_k not in (1, 2, 4, 8):
        raise ValueError('The source packing experiment supports top-k 1,2,4,8')
    output = gradient.new_empty((rows, gradient.shape[1]))
    _restore_input[(triton.cdiv(rows, 4), triton.cdiv(gradient.shape[1], 128))](gradient,
                                                                                inverse,
                                                                                output,
                                                                                rows,
                                                                                gradient.shape[1],
                                                                                top_k,
                                                                                BT=4,
                                                                                BH=128)
    return output
