# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from gdn_vendor import chunk_gated_delta_rule
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

modeling_qwen3_5_moe.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
