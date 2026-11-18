# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Fused EP gate with one quantization per original token before top-k packing."""
import torch
import torch.distributed as dist
from fused_gate_dispatch import FusedGateDispatch
from source_pack import pack_quantized_source, inverse_assignments, restore_input_gradient


class SourceGateDispatch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, group, hcom, sent_counts, received_counts, config, order, top_k):
        ctx.source_rows = x.shape[0]
        ctx.source_top_k = top_k
        ctx.source_inverse = inverse_assignments(order)
        if config is None:
            packed = x.index_select(0, order // top_k)
            return FusedGateDispatch.forward(ctx, packed, weight, group, hcom, sent_counts, received_counts, config)
        if config.input_dst_type_max is None:
            raise ValueError('Source-side scale must use an explicit quantization range')
        amax = torch.maximum(x.amax().float(), -x.amin().float()).reshape(1)
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=group)
        scale = (amax / config.input_dst_type_max).clamp_min(torch.finfo(torch.float32).tiny)
        payload = pack_quantized_source(x, order, top_k, scale)
        return FusedGateDispatch.forward(ctx,
                                         x,
                                         weight,
                                         group,
                                         hcom,
                                         sent_counts,
                                         received_counts,
                                         config,
                                         prepared_payload=payload,
                                         prepared_scale=scale)

    @staticmethod
    def backward(ctx, *gradients):
        packed_gradient = FusedGateDispatch.backward(ctx, *gradients)[0]
        dx = restore_input_gradient(packed_gradient, ctx.source_inverse, ctx.source_rows, ctx.source_top_k)
        return dx, None, None, None, None, None, None, None, None
