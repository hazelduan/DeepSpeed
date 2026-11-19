# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Dispatch quantized FC2 output gradients using the receiver's per-tensor scale."""
import torch
import torch_npu

from deepspeed import comm as dist


def fused_down_input_grad(grad, weight_payload, weight_scale, route, grad_range):
    """Compute FC2 dX without a receiver-side BF16 permutation buffer."""
    peers = grad.split(route.receive)
    zero = grad.new_zeros(())
    peer_max = torch.stack([torch.maximum(rows.amax(), -rows.amin()) if rows.numel() else zero
                            for rows in peers]).float()
    dist.all_reduce(peer_max, op=dist.ReduceOp.MAX, group=route.group)
    peer_scales = (peer_max / grad_range).clamp_min_(torch.finfo(torch.float32).tiny)
    local_rank = dist.get_rank(group=route.group)
    local_scale = peer_scales[local_rank:local_rank + 1]
    row_scales = peer_scales.new_empty(grad.shape[0])
    for index, rows in enumerate(row_scales.split(route.receive)):
        rows.copy_(peer_scales[index].expand_as(rows))
    payload = torch_npu.npu_quantize(grad.contiguous(),
                                     scales=row_scales,
                                     zero_points=None,
                                     dtype=torch_npu.hifloat8,
                                     axis=0,
                                     div_mode=True)
    dx, _, _ = torch_npu.npu_alltoallv_quant_gmm(payload,
                                                 weight_payload,
                                                 local_scale,
                                                 weight_scale.reshape(1),
                                                 route.hcom,
                                                 route.world_size,
                                                 route.receive_counts,
                                                 route.send_counts,
                                                 torch.bfloat16,
                                                 gmm_x_quant_mode=1,
                                                 gmm_weight_quant_mode=1,
                                                 gmm_x_dtype=torch_npu.hifloat8,
                                                 gmm_weight_dtype=torch_npu.hifloat8)
    return dx
