# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Record initial tensors in original expert-ID order across placement changes."""
import hashlib
import json
import os
from pathlib import Path

import torch
from lora_audit_fused_v2 import fingerprint


def digest(value):
    return hashlib.sha256(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def record_initial(audit):
    special = set()
    expert = {}
    routers = {}
    for name, layer in audit.layers:
        mapping = getattr(layer, '_system_new_to_original', torch.arange(layer.num_experts))
        local_ids = mapping[audit.rank * layer.num_local_experts:(audit.rank + 1) * layer.num_local_experts].tolist()
        for suffix, parameter in layer.experts.named_parameters():
            full_name = name + '.experts.' + suffix
            special.add(full_name)
            if suffix.startswith('lora'):
                sample = audit.initial[full_name]
                full = True
            elif os.environ.get('PRECISION_FULL_FINGERPRINTS') == '1':
                sample = parameter.detach().cpu()
                full = True
            else:
                rows, cols = parameter.shape[-2:]
                sample = torch.stack(
                    (parameter[:, 0, :32], parameter[:, rows // 2, cols // 2 - 16:cols // 2 + 16], parameter[:, -1,
                                                                                                             -32:]),
                    dim=1).detach().cpu()
                full = False
            expert[full_name] = dict(original_ids=local_ids,
                                     sha256=[digest(row) for row in sample],
                                     shape_per_expert=list(parameter.shape[1:]),
                                     full=full)
        inverse = torch.argsort(mapping)
        for suffix, parameter in layer.router.gate.named_parameters():
            full_name = name + '.router.gate.' + suffix
            special.add(full_name)
            value = parameter.detach().cpu().index_select(0, inverse)
            routers[full_name] = dict(sha256=digest(value), shape=list(value.shape), full=True)
    common_frozen = {
        name: fingerprint(p)
        for name, p in audit.model.named_parameters() if not p.requires_grad and name not in special
    }
    common_buffers = {name: fingerprint(p) for name, p in audit.model.named_buffers() if '._system_' not in name}
    common_trainable = {
        name: dict(sha256=digest(p), shape=list(p.shape))
        for name, p in audit.initial.items() if name not in special
    }
    expected_path = os.environ.get('PRECISION_CHECKPOINT_PROJECTION_HASHES')
    if expected_path:
        expected = json.loads(Path(expected_path).read_text())['checkpoint_tensors']
        mismatches = [
            name for name, value in expected.items() if name not in common_frozen or not common_frozen[name]['full']
            or common_frozen[name]['sha256'] != value['sha256']
        ]
        assert not mismatches, f'Checkpoint projection weights differ before training: {mismatches[:8]}'
    record = dict(
        rank=audit.rank,
        experts=expert,
        routers=routers,
        common_frozen=common_frozen,
        common_buffers=common_buffers,
        common_trainable=common_trainable,
        caveat=('All parameter values are fully hashed.' if os.environ.get('PRECISION_FULL_FINGERPRINTS') == '1' else
                'Expert base weights are sampled; adapter tensors and router weights are fully hashed.'))
    (audit.root / f'canonical_initial_rank{audit.rank}.json').write_text(json.dumps(record, indent=2))
    manifest_path = audit.root / f'manifest_rank{audit.rank}.json'
    manifest = json.loads(manifest_path.read_text())
    source = Path(__file__)
    manifest['source_sha256'][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))
