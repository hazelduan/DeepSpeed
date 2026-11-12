# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Opt-in NPU DP2 ZeRO save/load validation for Dense and grouped HiFloat8.

Run with a valid two-device Ascend rank table and the optional TorchAO HiFloat8 extension installed::

    torchrun --nproc_per_node=2 tests/unit/runtime/hifloat8_resume_validation.py \
        --output /tmp/hifloat8-resume

GroupedExperts are replicated for this DP2 lifecycle check. AutoEP expert
parallel routing is validated separately with a model integration run.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import torch_npu  # noqa: F401
from torch.utils.checkpoint import checkpoint
import deepspeed
import deepspeed.comm as dist
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from torchao.hifloat8 import HiFloat8GMM, HiFloat8Linear


class Model(torch.nn.Module):

    def __init__(self, activation_checkpointing=False):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.dense = torch.nn.Linear(128, 128, bias=False)
        self.experts = GroupedExperts(128, 128, 2, use_grouped_mm=False)
        with torch.no_grad():
            for p in self.parameters():
                p.normal_(std=0.02)

    def forward(self, x):
        if self.activation_checkpointing:
            return checkpoint(self._forward, x, use_reentrant=True)
        return self._forward(x)

    def _forward(self, x):
        counts = torch.tensor((0, x.shape[0]), device=x.device, dtype=torch.int64)
        return self.experts(self.dense(x), counts)


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(snapshot(v) for v in value)
    if isinstance(value, list):
        return [snapshot(v) for v in value]
    return copy.deepcopy(value)


def compare_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, LossScaler):
        assert type(a) is type(b)
        compare_nested(vars(a), vars(b))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare_nested(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare_nested(x, y)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module-recipes', action='store_true')
    parser.add_argument('--zero-stage', type=int, choices=(2, 3), default=2)
    parser.add_argument('--zero-init', action='store_true')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--freeze-first-expert', action='store_true')
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    if args.steps < 8 or args.steps % 2:
        parser.error('--steps must be an even number of microsteps, at least 8')
    if args.zero_init and args.zero_stage != 3:
        parser.error('--zero-init requires --zero-stage 3')
    rank = int(os.environ['RANK'])
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    deepspeed.init_distributed(dist_backend='hccl')
    torch.manual_seed(42)
    initial = Model().bfloat16().state_dict()
    config = {
        'train_micro_batch_size_per_gpu': 1,
        'gradient_accumulation_steps': 2,
        'bf16': {
            'enabled': True
        },
        'zero_optimization': {
            'stage': args.zero_stage,
            'overlap_comm': False,
            'reduce_bucket_size': 1000000,
        },
        'quantization': {
            'hifloat8': {
                'enabled': True,
                'module_name_patterns': ['dense', 'experts'],
                'expected_module_count': 2,
                'min_numel': 16384,
            }
        },
        'steps_per_print': 10000,
    }
    generator = torch.Generator().manual_seed(123 + rank)
    if args.zero_stage == 3:
        config['zero_optimization']['stage3_prefetch_bucket_size'] = 0
        config['zero_optimization']['stage3_param_persistence_threshold'] = 0
        config['zero_optimization']['stage3_max_reuse_distance'] = 0
    if args.module_recipes:
        config['quantization']['hifloat8'].pop('module_name_patterns')
        config['quantization']['hifloat8']['module_configs'] = {
            'dense': {
                'grad_dst_type_max': 15
            },
            'experts': {
                'grad_dst_type_max': 224
            },
        }
    batches = [torch.randn(32, 128, generator=generator).bfloat16().to('npu') for _ in range(args.steps)]
    for batch in batches:
        batch.requires_grad_(args.checkpoint)
    targets = [torch.randn(32, 128, generator=generator).bfloat16().to('npu') * 0.01 for _ in range(args.steps)]

    def create():
        if args.zero_init:
            with deepspeed.zero.Init(config_dict_or_path=config, dtype=torch.bfloat16, remote_device='cpu'):
                model = Model(args.checkpoint)
            with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=0):
                if rank == 0:
                    model.load_state_dict(initial)
        else:
            model = Model(args.checkpoint).bfloat16()
            model.load_state_dict(initial)
        if args.freeze_first_expert:
            model.experts.w1.requires_grad_(False)
        parameters = dict(model.named_parameters())
        state_keys = tuple(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        assert tuple(engine.module.state_dict()) == state_keys
        assert all(dict(engine.module.named_parameters())[name] is parameter for name, parameter in parameters.items())
        assert isinstance(engine.module.dense, HiFloat8Linear)
        assert isinstance(engine.module.experts.grouped_mm, HiFloat8GMM)
        if args.module_recipes:
            assert engine.module.dense.config.grad_dst_type_max == 15
            assert engine.module.experts.grouped_mm.config.grad_dst_type_max == 224
        else:
            assert vars(engine.module.dense.config) == vars(engine.module.experts.grouped_mm.config)
        return engine

    def model_snapshot(engine):
        if args.zero_stage == 3:
            with deepspeed.zero.GatheredParameters(engine.module.parameters()):
                return snapshot(engine.module.state_dict())
        return snapshot(engine.module.state_dict())

    step_times = []
    allocated_peaks = []
    reserved_peaks = []

    def step(engine, i):
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        started = time.perf_counter()
        batches[i].grad = None
        loss = (engine(batches[i]).float() - targets[i].float()).square().mean()
        assert bool(torch.isfinite(loss))
        engine.backward(loss)
        engine.step()
        torch.npu.synchronize()
        step_times.append(time.perf_counter() - started)
        allocated_peaks.append(torch.npu.max_memory_allocated())
        reserved_peaks.append(torch.npu.max_memory_reserved())
        return float(loss.detach())

    engine = create()
    losses = []
    grad_norms = []
    for i in range(args.steps):
        losses.append(step(engine, i))
        if i % 2:
            grad_norms.append(float(engine.get_global_grad_norm()))
            assert math.isfinite(grad_norms[-1])
        if i == 3:
            assert engine.global_steps == 2
            engine.save_checkpoint(str(args.output / 'checkpoint'), tag='step2', client_state={'marker': 2})
    final = model_snapshot(engine)
    assert final.keys() == initial.keys()
    assert all(value.shape == initial[name].shape for name, value in final.items())
    assert any(not torch.equal(value, initial[name]) for name, value in final.items())
    if args.freeze_first_expert:
        assert torch.equal(final['experts.w1'], initial['experts.w1'])
    optimizer_final = snapshot(engine.optimizer.state_dict())
    resumed = create()
    path, state = resumed.load_checkpoint(str(args.output / 'checkpoint'), tag='step2')
    assert path and state['marker'] == 2 and resumed.global_steps == 2
    resumed_losses = []
    resumed_norms = []
    for i in range(4, args.steps):
        resumed_losses.append(step(resumed, i))
        if i % 2:
            resumed_norms.append(float(resumed.get_global_grad_norm()))
    assert resumed_losses == losses[4:]
    assert resumed_norms == grad_norms[2:]
    compare_nested(final, model_snapshot(resumed))
    compare_nested(optimizer_final, resumed.optimizer.state_dict())
    assert resumed.global_steps == args.steps // 2
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        'rank': rank,
        'world_size': dist.get_world_size(),
        'zero_stage': args.zero_stage,
        'zero_init': args.zero_init,
        'frozen_first_expert': args.freeze_first_expert,
        'activation_checkpointing': args.checkpoint,
        'gradient_accumulation_steps': 2,
        'optimizer_steps': args.steps // 2,
        'losses': losses,
        'resume_losses': resumed_losses,
        'grad_norms': grad_norms,
        'resume_grad_norms': resumed_norms,
        'microstep_seconds': step_times[:args.steps],
        'training_allocated_peak_bytes': max(allocated_peaks[:args.steps]),
        'training_reserved_peak_bytes': max(reserved_peaks[:args.steps]),
        'final_parameter_hashes': {
            name: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in final.items()
        },
        'parameters_exact': True,
        'optimizer_exact': True,
        'same_policy_values': not args.module_recipes,
        'module_recipes': args.module_recipes
    }
    (args.output / f'result_rank{rank}.json').write_text(json.dumps(result, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
