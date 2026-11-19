# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Fused HiFloat8 FC2 with a rank-sized LoRA return and source-side B projection."""
from collections import Counter
from dataclasses import dataclass

import torch
import torch_npu

from deepspeed import comm as dist
from deepspeed.module_inject.auto_ep_layer import _AllToAllV
from deepspeed.ops.npu.grouped_mm import grouped_mm
from fc2_dx_dispatch import fused_down_input_grad
from torchao_npu.features.training.hifloat8.hifloat8_gmm import _quantize_grouped

COUNTS = Counter()


@dataclass
class ReturnRoute:
    group: object
    hcom: str
    world_size: int
    send: list
    receive: list
    send_counts: list
    receive_counts: list


class _FrozenDownReturn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, weight, route, config):
        if weight.requires_grad or config.compute_dtype not in (None, torch.bfloat16):
            raise ValueError("FC2 combine requires frozen BF16-backed HiFloat8 weights")
        inputs, input_scale = _quantize_grouped(hidden, config.input_dst_type_max)
        weights, weight_scale = _quantize_grouped(weight, config.weight_dst_type_max)
        output, _ = torch_npu.npu_quant_gmm_alltoallv(inputs,
                                                      weights.transpose(-1, -2),
                                                      input_scale.reshape(1),
                                                      weight_scale.reshape(1),
                                                      route.hcom,
                                                      route.world_size,
                                                      route.send_counts,
                                                      route.receive_counts,
                                                      torch.bfloat16,
                                                      gmm_x_quant_mode=1,
                                                      gmm_weight_quant_mode=1,
                                                      gmm_x_dtype=torch_npu.hifloat8,
                                                      gmm_weight_dtype=torch_npu.hifloat8)
        ctx.save_for_backward(weight, weights, weight_scale)
        ctx.route = route
        ctx.grad_range = config.grad_dst_type_max
        COUNTS["fused_down_return"] += 1
        return output

    @staticmethod
    def backward(ctx, grad):
        _, weights, weight_scale = ctx.saved_tensors
        dx = fused_down_input_grad(grad, weights, weight_scale, ctx.route, ctx.grad_range)
        COUNTS["fused_down_input_grad"] += 1
        COUNTS["down_grad_quantized_elements"] += grad.numel()
        return dx, None, None, None


class _GatherAdapterB(torch.autograd.Function):

    @staticmethod
    def forward(ctx, local_weight, route):
        ctx.group = route.group
        ctx.local_shape = local_weight.shape
        weights = local_weight.new_empty((route.world_size * local_weight.shape[0], *local_weight.shape[1:]))
        dist.all_gather_into_tensor(weights, local_weight.contiguous(), group=route.group)
        COUNTS["adapter_b_gather"] += 1
        return weights

    @staticmethod
    def backward(ctx, grad):
        # Sum partial source gradients in FP32 before returning the owning shard.
        partial = grad.float().contiguous()
        local_grad = partial.new_empty(ctx.local_shape)
        dist.reduce_scatter_tensor(local_grad, partial, group=ctx.group)
        COUNTS["adapter_b_reduce"] += 1
        return local_grad.to(grad.dtype), None


def fused_fc2(hidden, weight, adapter_a, adapter_b, adapter_scale, offsets, source_offsets, inverse, route, config):
    base = _FrozenDownReturn.apply(hidden, weight, route, config)
    inner = grouped_mm(hidden, adapter_a, offsets, trans_b=True).mul_(adapter_scale)
    inner = _AllToAllV.apply(route.group, inner.index_select(0, inverse), route.send, route.receive)
    adapter_b = _GatherAdapterB.apply(adapter_b, route)
    return base.add_(grouped_mm(inner, adapter_b, source_offsets, trans_b=True))


def combine_fc2(layer, hidden, offsets, inverse, plan, config, sent, received):
    source_offsets = layer._system_last_counts.cumsum(0, dtype=torch.int64)
    route = ReturnRoute(layer.ep_group, layer._fused_gate_hcom, layer.ep_size, plan.output_splits, plan.input_splits,
                        [n for source in received for n in source], [n for destination in sent for n in destination])
    adapter = layer.experts.lora2
    return fused_fc2(hidden, layer.experts.w2, adapter.a, adapter.b, adapter.scale, offsets, source_offsets, inverse,
                     route, config)
