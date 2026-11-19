# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Qualify the stock-BF16/system-HiFloat8 pair with complete initial controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_run(root, warmup):
    manifests = [read_json(root / f"manifest_rank{rank}.json") for rank in range(8)]
    completed = [read_json(root / f"complete_rank{rank}.json") for rank in range(8)]
    rows = [[json.loads(line) for line in (root / f"steps_rank{rank}.jsonl").read_text().splitlines()]
            for rank in range(8)]
    steps = len(rows[0])
    assert steps >= warmup + 11
    for rank in range(8):
        assert len(rows[rank]) == completed[rank]["steps"] == steps
        assert completed[rank]["all_adapters_updated"]
        assert completed[rank]["adapter_tensors"] == 576
        manifest = manifests[rank]
        assert manifest["layers"] == 48 and manifest["world_size"] == 8
        assert manifest["local_experts"] == 32 and manifest["top_k"] == 8
        assert manifest["full_initial_fingerprints"]
        assert manifest["hccl_op_expansion_mode"] == "default"
        assert manifest["config"]["zero_optimization"]["stage"] == 2
        for index, row in enumerate(rows[rank], 1):
            assert row["step"] == index and row["layers_executed"] == 48
            assert row["finite_adapter_gradients"] and row["supervised_tokens"] > 0
            assert len(row["losses"]) == 1
            assert all(math.isfinite(row[key]) for key in ("grad_norm", "step_s"))
            assert math.isfinite(row["losses"][0])
            if manifest["hifloat8"]:
                assert row["native_ops"].get("npu_grouped_matmul_hifloat8", 0) >= 48 * 3
                assert row["native_ops"].get("npu_alltoallv_quant_gmm", 0) >= 48
    durations = [max(rows[rank][step]["e2e_s"] for rank in range(8)) for step in range(warmup, steps)]
    assert all(math.isfinite(value) and value > 0 for value in durations)
    tokens = sum(rows[rank][step]["batch_tokens"] for rank in range(8) for step in range(warmup, steps))
    canonical = [read_json(root / f"canonical_initial_rank{rank}.json") for rank in range(8)]
    experts = {}
    for record in canonical:
        assert record["experts"] and record["common_frozen"]
        assert all(value["full"] for value in record["common_frozen"].values())
        for name, value in record["experts"].items():
            assert value["full"], "Sampled initial weights cannot qualify a comparison"
            for identity, digest in zip(value["original_ids"], value["sha256"], strict=True):
                key = (name, identity)
                assert key not in experts
                experts[key] = (digest, value["shape_per_expert"])
    metrics = {
        "root": str(root),
        "steps": steps,
        "warmup_steps": warmup,
        "measured_steps": steps - warmup,
        "measured_tokens": tokens,
        "mean_step_s": statistics.mean(durations),
        "median_step_s": statistics.median(durations),
        "tokens_per_s": tokens / sum(durations),
        "peak_allocated_gib": max(row["peak_allocated_bytes"] for rank in rows for row in rank) / 2**30,
        "peak_reserved_gib": max(row["peak_reserved_bytes"] for rank in rows for row in rank) / 2**30
    }
    return manifests, rows, canonical, experts, metrics


def relative(actual, reference):
    return abs(actual - reference) / max(abs(reference), 1e-8)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("bf16", type=Path)
    parser.add_argument("hifloat8", type=Path)
    parser.add_argument("--checkpoint-oracle", required=True, type=Path)
    parser.add_argument("--routing-profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", default=5, type=int)
    args = parser.parse_args()
    bm, br, bc, be, bmetrics = read_run(args.bf16, args.warmup)
    hm, hr, hc, he, hmetrics = read_run(args.hifloat8, args.warmup)
    assert bmetrics["steps"] == hmetrics["steps"]
    assert all(not item["hifloat8"] for item in bm)
    assert all(item["hifloat8"] for item in hm)
    profile_hash = hashlib.sha256(args.routing_profile.read_bytes()).hexdigest()
    for manifest in hm:
        assert manifest["routing_profile_sha256"] == profile_hash
        stack = manifest["system_stack"]
        assert stack["expert_placement"] == "profile-greedy"
        for key in ("payload_reuse", "gate_dispatch_fusion", "up_payload_reuse", "buffer_reuse",
                    "source_quant_before_duplication", "native_expert_sort", "fp32_route_gradient_restore"):
            assert stack[key], key
        if manifest.get("fc2_experiment"):
            assert manifest["production_offload"]["weights"] == 132
            assert manifest["fc2_experiment"]["fused_dx"] and manifest["fc2_experiment"]["native_weighted_restore"]
        else:
            assert stack["dense_linear_payload_cache_budget_bytes"] == 4 * 2**30
    mismatches = []
    if be != he:
        mismatches.append("original-ID expert parameters")
    oracle = read_json(args.checkpoint_oracle)["checkpoint_tensors"]
    assert len(oracle) == 72
    input_mismatches = []
    shape_keys = ("lora_rank", "lora_alpha", "lora_dropout", "micro_batch", "sequence_cap", "world_size",
                  "gradient_accumulation_steps", "gdn_vendor", "trainable_numel")
    for rank in range(8):
        for key in shape_keys:
            assert bm[rank][key] == hm[rank][key], (rank, key)
        for key in ("routers", "common_frozen", "common_buffers", "common_trainable"):
            if bc[rank][key] != hc[rank][key]:
                mismatches.append(f"rank{rank}:{key}")
        for label, records in (("bf16", bc), ("hifloat8", hc)):
            for name, expected in oracle.items():
                if records[rank]["common_frozen"].get(name, {}).get("sha256") != expected["sha256"]:
                    mismatches.append(f"{label}:rank{rank}:checkpoint:{name}")
        for b, h in zip(br[rank], hr[rank], strict=True):
            for key in ("batch_sha256", "batch_tokens", "supervised_tokens"):
                if b[key] != h[key]:
                    input_mismatches.append({"rank": rank, "step": b["step"], "key": key})
    assert bm[0]["dataset_sha256"] == hm[0]["dataset_sha256"]
    global_losses = [[
        statistics.mean(rows[rank][step]["losses"][0] for rank in range(8)) for step in range(bmetrics["steps"])
    ] for rows in (br, hr)]
    global_norms = [[
        statistics.mean(rows[rank][step]["grad_norm"] for rank in range(8)) for step in range(bmetrics["steps"])
    ] for rows in (br, hr)]
    loss_errors = [relative(h, b) for b, h in zip(*global_losses, strict=True)]
    norm_errors = [relative(h, b) for b, h in zip(*global_norms, strict=True)]
    per_rank = []
    for rank in range(8):
        losses = [relative(h["losses"][0], b["losses"][0]) for b, h in zip(br[rank], hr[rank], strict=True)]
        norms = [relative(h["grad_norm"], b["grad_norm"]) for b, h in zip(br[rank], hr[rank], strict=True)]
        per_rank.append({
            "rank": rank,
            "loss_relative_max": max(losses),
            "loss_relative_mean": statistics.mean(losses),
            "grad_norm_relative_max": max(norms),
            "grad_norm_relative_mean": statistics.mean(norms)
        })
    precision = {
        "initial_parameter_mismatches": mismatches,
        "input_mismatches": input_mismatches,
        "loss_relative_error_max": max(loss_errors),
        "loss_relative_error_mean": statistics.mean(loss_errors),
        "grad_norm_relative_error_max": max(norm_errors),
        "grad_norm_relative_error_mean": statistics.mean(norm_errors),
        "per_rank": per_rank,
        "bf16_global_loss": global_losses[0],
        "hifloat8_global_loss": global_losses[1],
        "bf16_global_grad_norm": global_norms[0],
        "hifloat8_global_grad_norm": global_norms[1]
    }
    passed = (not mismatches and not input_mismatches and max(loss_errors) <= .05
              and statistics.mean(loss_errors) <= .02 and max(norm_errors) <= .10
              and statistics.mean(norm_errors) <= .05)
    gain = (hmetrics["tokens_per_s"] / bmetrics["tokens_per_s"] - 1) * 100
    result = {
        "stock_bf16":
        bmetrics,
        "system_hifloat8":
        hmetrics,
        "shape": {
            key: bm[0][key]
            for key in shape_keys
        },
        "throughput_gain_percent":
        gain,
        "precision":
        precision,
        "routing_profile_sha256":
        profile_hash,
        "precision_gate_passed":
        passed,
        "joint_20_percent_gate_passed":
        passed and gain >= 20,
        "historical_32_28_percent_single_pair_gate_passed":
        passed and gain >= 32.28002546505242,
        "thresholds": {
            "loss_max": .05,
            "loss_mean": .02,
            "grad_norm_max": .10,
            "grad_norm_mean": .05
        },
        "limitations":
        "Short-run qualification only; optimizer updates are checked, but convergence and long-run accuracy are not established."
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
