# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Record actual post-AutoEP adapters, full-layer execution and comparable timings."""
import hashlib
import json
import os
from pathlib import Path
import time
from collections import Counter

import torch
import torch_npu
import deepspeed
import deepspeed.comm as dist
from swift import TrainerCallback, callbacks_map


class ExpertLoRAAudit(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        self.engine = self.trainer.model_wrapped
        self.model = self.engine.module
        self.model.enable_input_require_grads()
        self.rank = args.process_index
        self.root = Path(args.output_dir)
        self.layers = [(n, m) for n, m in self.model.named_modules() if isinstance(m, AutoEPMoELayer)]
        assert len(self.layers) == 48
        assert all(m.num_local_experts == 32 and m.top_k == 8 for _, m in self.layers)
        self.parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        assert len(self.parameters) == 48 * 12, len(self.parameters)
        assert all('.lora' in n or ('.shared_experts.' in n and n.endswith(('.a', '.b'))) for n in self.parameters)
        self.initial = {n: p.detach().cpu().clone() for n, p in self.parameters.items()}
        self.hits = Counter()
        self.handles = [m.register_forward_hook(lambda m, a, b, n=n: self.hits.update([n])) for n, m in self.layers]
        self.gradient_checks = []
        self.handles.extend(p.register_hook(self.check_gradient(n)) for n, p in self.parameters.items())
        original_loss = self.trainer.compute_loss

        def compute_loss(model, inputs, *a, **kw):
            digest = hashlib.sha256()
            mask = inputs.get('attention_mask')
            self.batch_tokens += int(
                mask.sum()) if isinstance(mask, torch.Tensor) and mask.ndim == 2 else inputs['input_ids'].numel()
            self.supervised_tokens += int((inputs['labels'] != -100).sum())
            for name in ('input_ids', 'labels', 'attention_mask'):
                value = inputs.get(name)
                if isinstance(value, torch.Tensor):
                    digest.update(name.encode())
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            self.batch_hashes.append(digest.hexdigest())
            result = original_loss(model, inputs, *a, **kw)
            loss = result[0] if isinstance(result, tuple) else result
            assert bool(torch.isfinite(loss)), 'Nonfinite loss'
            self.losses.append(float(loss.detach()))
            return result

        self.trainer.compute_loss = compute_loss
        # This keeps the exact causal loss while bounding its FP32 temporaries.
        from deepspeed.runtime.chunked_cross_entropy import install_chunked_causal_lm_loss
        install_chunked_causal_lm_loss(self.model, block_rows=64, backend='torch')
        self.counts = Counter()
        for name in ('npu_grouped_matmul', 'npu_quant_matmul', 'npu_dynamic_quant'):
            original = getattr(torch_npu, name)

            def counted(*a, _original=original, _name=name, **kw):
                key = _name
                if kw.get('x_dtype', kw.get('x1_dtype')) == torch_npu.hifloat8:
                    key += '_hifloat8'
                self.counts[key] += 1
                return _original(*a, **kw)

            setattr(torch_npu, name, counted)
        manifest = {
            'rank': self.rank,
            'layers': 48,
            'local_experts': 32,
            'top_k': 8,
            'lora_rank': int(os.environ['EXPERT_LORA_RANK']),
            'lora_alpha': 2 * int(os.environ['EXPERT_LORA_RANK']),
            'lora_dropout': 0,
            'torch_npu': torch_npu.__version__,
            'world_size': args.world_size,
            'micro_batch': args.per_device_train_batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'sequence_cap': int(os.environ['SEQUENCE_CAP']),
            'gdn_vendor': os.environ.get('GDN_VENDOR'),
            'hccl_op_expansion_mode': os.environ.get('HCCL_OP_EXPANSION_MODE', 'default'),
            'full_initial_fingerprints': os.environ.get('PRECISION_FULL_FINGERPRINTS') == '1',
            'deepspeed': deepspeed.__file__,
            'torch': torch.__version__,
            'hifloat8': self.engine.hifloat8_enabled(),
            'config': self.engine._config._param_dict,
            'trainable_numel': sum(p.numel() for p in self.parameters.values()),
            'model_numel_local': sum(p.numel() for p in self.model.parameters()),
            'trainable': {
                n: list(p.shape)
                for n, p in self.parameters.items()
            },
            'adapter_sha256': {
                n: hashlib.sha256(p.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                for n, p in self.initial.items()
            },
            'source_sha256': {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(__file__).with_name('expert_lora_plugin.py'),
                          Path(os.environ['TRAIN_NODE_SOURCE']))
            }
        }
        if self.rank == 0:
            dataset = Path(os.environ['HIF8_PERF_DATASET'])
            with dataset.open('rb') as stream:
                manifest['dataset_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest['source_sha256'].update({
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(deepspeed.__file__).parent / 'moe/ep_kernels.py',
                      Path(deepspeed.__file__).parent / 'module_inject/auto_ep_layer.py',
                      Path(os.environ['HIF8_PERF_GDN_PROVENANCE']))
        })
        (self.root / f'manifest_rank{self.rank}.json').write_text(json.dumps(manifest, indent=2, default=str))
        self.previous_end = None

    def check_gradient(self, name):

        def capture(grad):
            self.gradient_checks.append((name, torch.isfinite(grad).all()))
            return grad

        return capture

    def on_step_begin(self, args, state, control, **kwargs):
        dist.barrier()
        torch.npu.synchronize()
        self.begin = time.perf_counter()
        self.gradient_checks.clear()
        self.hits.clear()
        self.batch_hashes = []
        self.losses = []
        self.batch_tokens = self.supervised_tokens = 0
        self.counts.clear()

    def on_step_end(self, args, state, control, **kwargs):
        torch.npu.synchronize()
        dist.barrier()
        end = time.perf_counter()
        assert set(self.hits) == {n for n, _ in self.layers}, 'Some model layers did not execute'
        assert {n for n, _ in self.gradient_checks} == set(self.parameters), 'Missing adapter gradients'
        assert bool(torch.stack([v for _, v in self.gradient_checks]).all()), 'Nonfinite adapter gradients'
        assert self.supervised_tokens > 0
        if self.engine.hifloat8_enabled():
            assert self.counts['npu_grouped_matmul_hifloat8'] >= 48 * 3
        row = {
            'step': state.global_step,
            'rank': self.rank,
            'losses': self.losses,
            'step_s': end - self.begin,
            'e2e_s': None if self.previous_end is None else end - self.previous_end,
            'batch_tokens': self.batch_tokens,
            'supervised_tokens': self.supervised_tokens,
            'batch_sha256': self.batch_hashes,
            'layers_executed': len(self.hits),
            'finite_adapter_gradients': True,
            'grad_norm': float(self.engine.get_global_grad_norm()),
            'peak_allocated_bytes': torch.npu.max_memory_allocated(),
            'peak_reserved_bytes': torch.npu.max_memory_reserved(),
            'native_ops': dict(self.counts)
        }
        with (self.root / f'steps_rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        self.previous_end = end
        if state.global_step >= int(os.environ['MODEL_STEPS']):
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        changed = {n: bool((p.detach().cpu() != self.initial[n]).any()) for n, p in self.parameters.items()}
        # A starts receiving gradients after the zero-initialized B has updated.
        assert all(changed.values()), [n for n, value in changed.items() if not value]
        (self.root / f'complete_rank{self.rank}.json').write_text(
            json.dumps({
                'steps': state.global_step,
                'all_adapters_updated': True,
                'adapter_tensors': len(changed)
            }))


callbacks_map['expert_lora_audit'] = ExpertLoRAAudit
