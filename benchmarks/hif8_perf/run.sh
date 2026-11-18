#!/usr/bin/env bash
set -euo pipefail

mode=${1:?Usage: run.sh bf16|hifloat8 OUTPUT_DIR [STEPS]}
output=${2:?Provide a fresh output directory}
steps=${3:-20}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)

: "${HIF8_PERF_MODEL:?Set HIF8_PERF_MODEL to the local Qwen3.5-122B-A10B checkpoint}"
: "${HIF8_PERF_DATASET:?Set HIF8_PERF_DATASET to the conversation SFT JSONL}"
: "${HIF8_PERF_SWIFT_SFT:?Set HIF8_PERF_SWIFT_SFT to the Swift sft.py entry point}"
: "${HIF8_PERF_GDN_PROVENANCE:?Set HIF8_PERF_GDN_PROVENANCE to the GDN provider provenance JSON}"
: "${PRECISION_CHECKPOINT_PROJECTION_HASHES:?Set the checkpoint projection hash oracle}"

plugins=("$here/expert_lora_plugin.py")
checkpoint_args=()
case "$mode" in
    bf16)
        export HCCL_BUFFSIZE=1024
        plugins+=("$here/lora_audit_stock.py")
        ;;
    hifloat8)
        : "${HIF8_PERF_AO_ROOT:?Set HIF8_PERF_AO_ROOT to the matching TorchAO Ascend checkout}"
        : "${SYSTEM_ROUTING_PROFILE:?Set SYSTEM_ROUTING_PROFILE to the full-model routing profile JSON}"
        export HCCL_BUFFSIZE=256 DENSE_LINEAR_CACHE_BYTES=4294967296 HIF8_CE_BLOCK_ROWS=512
        checkpoint_args=(--gradient_checkpointing_kwargs '{"use_reentrant": false}')
        plugins+=("$here/fused_gate_lora_plugin.py" "$here/system_stack.py" "$here/system_source_stack.py"
                  "$here/lora_audit_source_cached.py")
        ;;
    *)
        echo "Mode must be bf16 or hifloat8" >&2
        exit 2
        ;;
esac

export PYTHONPATH="$here:$repo${HIF8_PERF_AO_ROOT:+:$HIF8_PERF_AO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DS_ACCELERATOR=npu OMP_NUM_THREADS=4 PYTHONHASHSEED=42 HCCL_CONNECT_TIMEOUT=180
export PRECISION_FULL_FINGERPRINTS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HIF8_MEMORY_FRACTION=.95 HIF8_MIN_FREE_GIB=80
export EXPERT_LORA_ENABLE=1 EXPERT_LORA_RANK=8 GDN_VENDOR=1 MODEL_STEPS="$steps" SEQUENCE_CAP=256
export TRAIN_NODE_SOURCE="$here/run.sh"
unset RANK_TABLE_FILE HCCL_OP_EXPANSION_MODE

output=$(realpath -m -- "$output")
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
export HIF8_RUN_OUTPUT="$output"
python - "$here/hifloat8.json" "$output/deepspeed.json" "$mode" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
config["quantization"]["hifloat8"]["enabled"] = sys.argv[3] == "hifloat8"
Path(sys.argv[2]).write_text(json.dumps(config, indent=2) + "\n")
PY

plugins+=("$here/gdn_provider.py" "$here/resource_step_windows.py")
if [[ "$mode" == hifloat8 ]]; then
    plugins+=("$here/native_restore_input.py" "$here/hif8_loss_blocks.py")
fi
plugins+=("$here/post_training_integrity.py")

cmd=(python "$here/launch_ranks.py" --node-rank 0 --nnodes 1 --local-world 8
    --master-addr "${HIF8_PERF_MASTER_ADDR:-127.0.0.1}" --master-port "${HIF8_PERF_MASTER_PORT:-47411}"
    --log-dir "$output/workers" "$here/train_entry.py" "$HIF8_PERF_SWIFT_SFT"
    --model "$HIF8_PERF_MODEL" --model_type qwen3_5_moe
    --use_hf true --check_model false --dataset "$HIF8_PERF_DATASET" --split_dataset_ratio 0
    --tuner_type full --freeze_parameters_regex '.*'
    --trainable_parameters_regex '^model\.language_model\.layers\.[0-9]+\.mlp\.(?:experts\.(?:gate_up_proj|down_proj)|shared_expert\.(?:gate_proj|up_proj|down_proj)\.weight)$'
    --torch_dtype bfloat16 --bf16 true --fp16 false --attn_impl eager --experts_impl eager
    --deepspeed "$output/deepspeed.json" --enable_npu_model_patch false --device_map cpu
    --per_device_train_batch_size 128 --gradient_accumulation_steps 1 --gradient_checkpointing true
    "${checkpoint_args[@]}" --max_length 256 --max_steps "$steps" --learning_rate 1e-5 --optim adamw_torch
    --lr_scheduler_type constant --warmup_ratio 0 --weight_decay 0 --max_grad_norm 1
    --seed 42 --data_seed 42 --dataset_shuffle true --lazy_tokenize false --load_from_cache_file false
    --dataset_num_proc 1 --dataloader_num_workers 0 --dataloader_persistent_workers false --dataloader_drop_last true
    --truncation_strategy right --use_logits_to_keep true --logging_steps 1 --logging_first_step true
    --eval_strategy no --report_to none --disable_tqdm true --include_num_input_tokens_seen true
    --add_version false --external_plugins "${plugins[@]}" --callbacks expert_lora_audit
    --output_dir "$output" --save_strategy no)
printf '%q ' "${cmd[@]}" > "$output/command.txt"
"${cmd[@]}" > "$output/launcher.log" 2>&1
