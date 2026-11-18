# HiFloat8 AutoEP benchmark

This benchmark exercises the full 48-layer Qwen3.5-122B-A10B on one eight-NPU
node with ZeRO-2, EP8, expert LoRA rank 8, micro batch 128 and sequence cap 256.
The original BF16 path and the optimized HiFloat8 path use identical batches
and the same checkpoint. HiFloat8 uses profile-based expert placement,
source-side quantization, fused dispatch and GMM, and native Top-K gradient
restoration. The optimized arm caches frozen Linear payloads and uses CE512
and a 256 MiB HCCL buffer.

Use the companion `hif8_perf` checkout of
[TorchAO Ascend](https://gitcode.com/Ascend/torchao-ascend). Its base is
`539dd693bd15e05563a35e21487cb8cbef850c51`. The GMM patch keeps the forward
weight payload for the input-gradient matmul.

Activate the experiment environment before running the launcher. The reference
environment used Ascend 950DT, Torch 2.12, TorchNPU 2.12.0.post2, Transformers
5.16.1 and CANN 9.2 beta. Put ms-swift, Triton Ascend and the model-compatible
`gdn_vendor` provider on the environment's Python path.

| Variable | Input |
| --- | --- |
| `HIF8_PERF_MODEL` | Local Qwen3.5-122B-A10B checkpoint |
| `HIF8_PERF_DATASET` | Conversation SFT JSONL |
| `HIF8_PERF_SWIFT_SFT` | ms-swift `swift/cli/sft.py` |
| `HIF8_PERF_AO_ROOT` | Companion TorchAO Ascend source root |
| `HIF8_PERF_GDN_PROVENANCE` | Provenance JSON for the GDN provider |
| `PRECISION_CHECKPOINT_PROJECTION_HASHES` | JSON containing the 72 checkpoint projection digests |
| `SYSTEM_ROUTING_PROFILE` | Per-layer routing counts in original expert order |

Run the two arms sequentially on the same node.

```sh
bash benchmarks/hif8_perf/run.sh bf16 /path/to/results/bf16 20
bash benchmarks/hif8_perf/run.sh hifloat8 /path/to/results/hifloat8 20
python benchmarks/hif8_perf/compare_revalidation_pair.py \
  /path/to/results/bf16 /path/to/results/hifloat8 \
  --checkpoint-oracle "$PRECISION_CHECKPOINT_PROJECTION_HASHES" \
  --routing-profile "$SYSTEM_ROUTING_PROFILE" \
  --output /path/to/results/comparison.json
```

Timing excludes the first five steps and uses the slowest rank for each
measured step. The reference implementation measured 22.53 seconds per step
for BF16 and 18.34–18.36 seconds for HiFloat8. Both runs passed the short loss,
gradient norm and frozen-weight checks. The dataset, routing profile, checkpoint
and environment are supplied by the experiment operator.
