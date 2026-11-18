# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Experimental frozen gate projection fused with EP dispatch.

The BF16 path preserves dispatched LoRA inputs exactly. The HiFloat8 path
uses one EP-wide input scale and passes the decoded payload to LoRA; this is
an explicit quantization-policy change, not a transparent replacement of
receiver-local scaling. Only frozen base weights and first-order gradients
are supported. Routing and adapter parameters remain framework-owned.
"""
import importlib
from collections import Counter

import torch
import torch_npu
import torch.distributed as dist
import triton
import triton.language as tl

PAYLOAD_STATISTICS = Counter()


@triton.jit
def _inverse_indices(metadata, output, segments: tl.constexpr, BLOCK: tl.constexpr):
    segment = tl.program_id(0)
    length = tl.load(metadata + segment)
    source = tl.load(metadata + segments + segment).to(tl.int64)
    target = tl.load(metadata + 2 * segments + segment).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    for start in range(0, length, BLOCK):
        rows = start + offsets
        tl.store(output + source + rows, target + rows, mask=rows < length)


def exchange(x, group, send, recv):
    y = x.new_empty((sum(recv), x.shape[1]))
    dist.all_to_all_single(y, x.contiguous(), output_split_sizes=recv, input_split_sizes=send, group=group)
    return y


def inverse_permutation(received_counts, device):
    """Map source-major positions to expert-major positions without padding."""
    counts = torch.tensor(received_counts, dtype=torch.int64)
    world, experts = counts.shape
    expert_start = counts.sum(0).cumsum(0) - counts.sum(0)
    offsets = expert_start.unsqueeze(0) + counts.cumsum(0) - counts
    if torch.device(device).type == 'npu':
        # Only the small count matrix crosses the host/device boundary. Avoid
        # constructing and copying a million-element index vector on the CPU.
        starts = counts.flatten().cumsum(0) - counts.flatten()
        metadata = torch.stack((counts.flatten(), starts, offsets.flatten())).to(device)
        result = torch.empty(int(counts.sum()), device=device, dtype=torch.int64)
        _inverse_indices[(world * experts, )](metadata, result, world * experts, BLOCK=256)
        return result
    pieces = [
        torch.arange(int(offsets[src, expert]), int(offsets[src, expert] + counts[src, expert]))
        for src in range(world) for expert in range(experts)
    ]
    return torch.cat(pieces).to(device)


def bf16_gmm(x, weight, offsets):
    return torch_npu.npu_grouped_matmul([x.contiguous()], [weight],
                                        group_list=offsets,
                                        group_list_type=0,
                                        split_item=3,
                                        group_type=0,
                                        output_dtype=torch.bfloat16)[0]


def frozen_dx(gradient, weight, offsets, config, weight_scale, weight_payload=None):
    if config is None:
        return bf16_gmm(gradient, weight, offsets)
    module = importlib.import_module('torchao_npu.features.training.hifloat8.hifloat8_gmm')
    gq, gs = module._quantize_grouped(gradient, config.grad_dst_type_max)
    if getattr(config, 'reuse_quantized_weight', False):
        if weight_payload is None:
            raise RuntimeError('The requested forward weight payload was not retained')
        wq, ws = weight_payload, weight_scale
        PAYLOAD_STATISTICS['fused_weight_payload_reuses'] += 1
    else:
        wq, ws = module._quantize_grouped(weight, None, scale=weight_scale)
    return module._grouped_mm(gq, wq, gs, ws, offsets, group_type=0, output_dtype=torch.bfloat16)


class PrequantizedFrozenGMM(torch.autograd.Function):
    """Consume the dispatched bytes directly; BF16 tokens carry the STE gradient."""

    @staticmethod
    def forward(ctx, tokens, payload, input_scale, weight, offsets, config):
        if weight.requires_grad or payload.shape != tokens.shape:
            raise ValueError('Expected a frozen projection and matching dispatched payload')
        module = importlib.import_module('torchao_npu.features.training.hifloat8.hifloat8_gmm')
        wq, ws = module._quantize_grouped(weight, config.weight_dst_type_max)
        output = module._grouped_mm(payload,
                                    wq.transpose(-1, -2),
                                    input_scale,
                                    ws,
                                    offsets,
                                    group_type=0,
                                    output_dtype=torch.bfloat16)
        saved_payload = wq if getattr(config, 'reuse_quantized_weight', False) else wq.new_empty(0)
        ctx.save_for_backward(weight, offsets, ws, saved_payload)
        ctx.config = config
        return output

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Prequantized frozen GMM supports first-order gradients only')
        weight, offsets, ws, payload = ctx.saved_tensors
        dx = (frozen_dx(gradient, weight, offsets, ctx.config, ws, payload) if gradient.shape[0] else weight.new_empty(
            (0, weight.shape[-1])))
        return dx, None, None, None, None, None


class FusedGateDispatch(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                x,
                weight,
                group,
                hcom,
                sent_counts,
                received_counts,
                config,
                prepared_payload=None,
                prepared_scale=None):
        if weight.requires_grad:
            raise ValueError('Fused gate dispatch currently requires frozen base weights')
        if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
            raise ValueError('The experiment requires BF16 model storage')
        world = len(received_counts)
        send = [sum(row) for row in sent_counts]
        recv = [sum(row) for row in received_counts]
        sc = [v for row in sent_counts for v in row]
        rc = [v for row in received_counts for v in row]
        ctx.group, ctx.send, ctx.recv = group, send, recv
        ctx.config = config
        counts = torch.tensor(received_counts, device=x.device).sum(0)
        offsets = counts.cumsum(0).to(torch.int64)
        inverse = inverse_permutation(received_counts, x.device)
        weight_scale = None
        tokens_q = xs = None
        if config is None:
            gate, _, tokens = torch_npu.npu_alltoallv_gmm(x,
                                                          weight,
                                                          hcom,
                                                          world,
                                                          sc,
                                                          rc,
                                                          trans_gmm_weight=True,
                                                          permute_out_flag=True)
        else:
            module = importlib.import_module('torchao_npu.features.training.hifloat8.hifloat8_gmm')
            if prepared_payload is None:
                # Two reductions avoid allocating a full abs(x) activation buffer.
                amax = torch.maximum(x.amax().float(), -x.amin().float()).reshape(1)
                dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=group)
                if config.input_dst_type_max is None:
                    raise ValueError('The experimental EP-wide quantization scale must be explicit')
                xs = (amax / config.input_dst_type_max).clamp_min(torch.finfo(torch.float32).tiny)
                xq = torch_npu.npu_quantize(x,
                                            scales=xs,
                                            zero_points=None,
                                            dtype=torch_npu.hifloat8,
                                            axis=1,
                                            div_mode=True)
            else:
                if prepared_scale is None or prepared_payload.shape != (sum(send), x.shape[1]):
                    raise ValueError('Prepared source payload and routing counts do not align')
                xq, xs = prepared_payload, prepared_scale
            wq, weight_scale = module._quantize_grouped(weight, config.weight_dst_type_max)
            gate, _, tokens_q = torch_npu.npu_alltoallv_quant_gmm(xq,
                                                                  wq.transpose(-1, -2).contiguous(),
                                                                  xs,
                                                                  weight_scale,
                                                                  hcom,
                                                                  world,
                                                                  sc,
                                                                  rc,
                                                                  torch.bfloat16,
                                                                  permute_out_flag=True,
                                                                  gmm_x_quant_mode=1,
                                                                  gmm_weight_quant_mode=1,
                                                                  gmm_x_dtype=torch_npu.hifloat8,
                                                                  gmm_weight_dtype=torch_npu.hifloat8)
            # HiFloat8 bytes are NOT IEEE FP8 and must use the native dtype tag.
            tokens = torch_npu.npu_dtype_cast(tokens_q, torch.bfloat16, input_dtype=torch_npu.hifloat8).mul_(xs)
        saved_payload = (wq if config is not None and getattr(config, 'reuse_quantized_weight', False) else
                         x.new_empty(0, dtype=torch.uint8))
        ctx.save_for_backward(weight, offsets, inverse, weight_scale if weight_scale is not None else x.new_empty(0),
                              saved_payload)
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(inverse, *((tokens_q, xs) if tokens_q is not None else ()))
        return gate, tokens, tokens_q, xs, inverse

    @staticmethod
    def backward(ctx, gate_gradient, token_gradient, _payload_gradient, _scale_gradient, _inverse_gradient):
        if torch.is_grad_enabled():
            raise RuntimeError('Fused gate dispatch supports first-order gradients only')
        weight, offsets, inverse, weight_scale, payload = ctx.saved_tensors
        dx = None
        if gate_gradient is not None and gate_gradient.shape[0]:
            dx = frozen_dx(gate_gradient, weight, offsets, ctx.config, weight_scale, payload)
        if token_gradient is not None:
            dx = token_gradient if dx is None else dx.add_(token_gradient)
        if dx is None:
            dx = weight.new_zeros((sum(ctx.recv), weight.shape[-1]))
        dx = exchange(dx.index_select(0, inverse), ctx.group, ctx.recv, ctx.send)
        return (dx, None, None, None, None, None, None) + (None, ) * (len(ctx.needs_input_grad) - 7)
