# HiFloat8 AutoEP benchmark

This benchmark exercises the full 48-layer Qwen3.5-122B-A10B on one eight-NPU
node with ZeRO-2, EP8, expert LoRA rank 8, micro batch 128 and sequence cap 256.
The original BF16 path and the optimized HiFloat8 path use identical batches
and the same checkpoint. HiFloat8 uses profile-based expert placement,
source-side quantization, fused dispatch and GMM, and native Top-K gradient
restoration. FC2 fuses the base projection with the return and applies LoRA B
on the source rank. The backward uses quantized dispatch with the receiver's
scale. The optimized arm uses native weighted Top-K restore, frozen Linear
host offload, CE512 and a 256 MiB HCCL buffer.

Use the companion `hif8_perf` checkout of
[TorchAO Ascend](https://gitcode.com/hazeldxq/torchao_npu/tree/hif8_perf) at
`a0ead21f6da43f64578b4c776fd79c4d090dce9e`. That revision supports frozen
Linear host offload and keeps forward GMM payloads for the input-gradient
matmul.

Activate the experiment environment before running the launcher. The reference
environment used Ascend 950DT, Torch 2.12, TorchNPU 2.12.0.post2, Transformers
5.16.1 and CANN 9.2 beta. Put ms-swift, Triton Ascend and the model-compatible
`gdn_vendor` provider on the environment's Python path.

| Variable | Input |
| --- | --- |
| `HIF8_PERF_MODEL` | Local Qwen3.5-122B-A10B checkpoint |
| `HIF8_PERF_DATASET` | Conversation SFT JSONL |
| `HIF8_PERF_SWIFT_SFT` | ms-swift `swift/cli/sft.py` |
| `HIF8_PERF_BF16_DS_ROOT` | Historical F1 BF16 runtime for the matched comparison |
| `HIF8_PERF_AO_ROOT` | Companion TorchAO Ascend source root |
| `HIF8_PERF_GDN_PROVENANCE` | Provenance JSON for the GDN provider |
| `PRECISION_CHECKPOINT_PROJECTION_HASHES` | JSON containing the 72 checkpoint projection digests |
| `SYSTEM_ROUTING_PROFILE` | Per-layer routing counts in original expert order |

Run the two arms sequentially on the same node.

```sh
bash benchmarks/hif8_perf/run.sh bf16 /path/to/results/bf16 16
bash benchmarks/hif8_perf/run.sh hifloat8 /path/to/results/hifloat8 16
python benchmarks/hif8_perf/compare_revalidation_pair.py \
  /path/to/results/bf16 /path/to/results/hifloat8 \
  --checkpoint-oracle "$PRECISION_CHECKPOINT_PROJECTION_HASHES" \
  --routing-profile "$SYSTEM_ROUTING_PROFILE" \
  --output /path/to/results/comparison.json
```

Timing excludes the first five steps and uses the slowest rank for each
measured step. The 16-step source experiment measured 22.504 seconds per step
for the historical F1 BF16 runtime and 17.267 seconds per step for HiFloat8
on the same eight Ascend 950DT devices, a 30.33% throughput gain. The loss,
gradient norm and frozen-weight checks passed. The dataset, routing profile,
checkpoint and environment are supplied by the experiment operator.
