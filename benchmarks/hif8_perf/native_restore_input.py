# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Restore top-k input derivatives with the validated native FP32 reduction."""
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
from swift import callbacks_map
import source_gate_dispatch

BaseAudit = callbacks_map['expert_lora_audit']


class NativeRestoreAudit(BaseAudit):

    def on_train_begin(self, *args, **kwargs):
        super().on_train_begin(*args, **kwargs)
        assert all(hasattr(layer, 'experimental_source_dispatch') for _, layer in self.layers)
        self.original_restore = source_gate_dispatch.restore_input_gradient
        self.restore_ones = {}
        self.native_restore_calls = 0

        def restore(gradient, inverse, rows, top_k):
            if rows == 0:
                return gradient.new_empty((0, gradient.shape[1]))
            if gradient.dtype != torch.bfloat16 or rows * top_k >= 2**31:
                return self.original_restore(gradient, inverse, rows, top_k)
            assert top_k in (1, 2, 4, 8)
            assert gradient.shape[0] == inverse.numel() == rows * top_k
            key = (gradient.device.index, rows, top_k)
            if key not in self.restore_ones:
                self.restore_ones[key] = torch.ones(rows, top_k, device=gradient.device, dtype=gradient.dtype)
            # finalize_routing addresses assignments in choice-major order.
            mapping = inverse.reshape(rows, top_k).T.contiguous().reshape(-1).int()
            self.native_restore_calls += 1
            return torch_npu.npu_moe_finalize_routing(gradient, None, None, None, self.restore_ones[key], mapping,
                                                      None, 0)

        source_gate_dispatch.restore_input_gradient = restore
        test = torch.linspace(-.25, .25, 17 * 8 * 256, device='npu', dtype=torch.bfloat16).reshape(17 * 8, 256)
        inverse = torch.arange(17 * 8, device='npu', dtype=torch.int64).mul(7).remainder(17 * 8)
        expected = test.index_select(0, inverse).reshape(17, 8, 256).float().sum(1).bfloat16()
        actual = restore(test, inverse, 17, 8)
        error = float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))
        assert error <= 2e-6, error
        path = self.root / f'manifest_rank{self.rank}.json'
        manifest = json.loads(path.read_text())
        manifest['native_input_gradient_restore'] = {
            'enabled': True,
            'primitive': 'npu_moe_finalize_routing',
            'fp32_oracle_nrmse': error,
            'index_dtype': 'int32'
        }
        manifest['source_sha256'][__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2))

    def on_step_begin(self, *args, **kwargs):
        self.native_restore_calls = 0
        return super().on_step_begin(*args, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        assert self.native_restore_calls == 48, self.native_restore_calls
        result = super().on_step_end(args, state, control, **kwargs)
        with (self.root / f'native_restore_rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': state.global_step, 'calls': self.native_restore_calls}) + '\n')
        return result

    def on_train_end(self, *args, **kwargs):
        try:
            return super().on_train_end(*args, **kwargs)
        finally:
            source_gate_dispatch.restore_input_gradient = self.original_restore
            self.restore_ones.clear()


callbacks_map['expert_lora_audit'] = NativeRestoreAudit
