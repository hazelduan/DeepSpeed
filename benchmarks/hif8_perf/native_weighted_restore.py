# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Use Ascend's differentiable weighted restore for Qwen AutoEP output."""
import torch
import torch_npu

from deepspeed.module_inject import auto_ep_layer
from fc2_combine import COUNTS
from source_pack import inverse_assignments

_original_combine = auto_ep_layer.combine_from_routed


def combine_from_routed(expert_output, top_scores, token_indices_sorted, top_k, score_apply, combine_impl, shape):
    num_assignments = shape[0] * shape[1] * top_k
    if (score_apply != "post" or combine_impl != "weighted_sum" or expert_output.device.type != "npu"
            or expert_output.dtype != torch.bfloat16 or top_scores.dtype != expert_output.dtype
            or not 0 < num_assignments < 2**31 or expert_output.shape[0] != num_assignments
            or token_indices_sorted.numel() != num_assignments):
        return _original_combine(expert_output, top_scores, token_indices_sorted, top_k, score_apply, combine_impl,
                                 shape)
    inverse = inverse_assignments(token_indices_sorted).int()
    output = torch_npu.npu_moe_token_unpermute(expert_output, inverse, probs=top_scores.contiguous())
    COUNTS["native_weighted_restore"] += 1
    return output.reshape(shape)


def install():
    auto_ep_layer.combine_from_routed = combine_from_routed
