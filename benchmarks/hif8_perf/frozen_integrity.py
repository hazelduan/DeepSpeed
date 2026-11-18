# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Verify every frozen parameter after training, outside the measured steps."""
import json

import torch
from canonical_provenance import digest
from lora_audit_fused_v2 import fingerprint


def verify_frozen_weights(audit):
    torch.npu.synchronize()
    initial = json.loads((audit.root / f'canonical_initial_rank{audit.rank}.json').read_text())
    parameters = dict(audit.model.named_parameters())
    covered = set(initial['common_frozen']) | set(initial['routers'])
    changed = [
        name for name, expected in initial['common_frozen'].items() if fingerprint(parameters[name]) != expected
    ]
    expert_slices = 0
    for name, layer in audit.layers:
        mapping = getattr(layer, '_system_new_to_original', torch.arange(layer.num_experts))
        inverse = torch.argsort(mapping)
        for suffix, parameter in layer.router.gate.named_parameters():
            key = name + '.router.gate.' + suffix
            actual = digest(parameter.detach().cpu().index_select(0, inverse))
            if actual != initial['routers'][key]['sha256']:
                changed.append(key)
        for suffix in ('w1', 'w2', 'w3'):
            key = name + '.experts.' + suffix
            covered.add(key)
            value = getattr(layer.experts, suffix).detach().cpu()
            actual = [digest(expert) for expert in value]
            if actual != initial['experts'][key]['sha256']:
                changed.append(key)
            expert_slices += len(actual)
    frozen = {name for name, parameter in parameters.items() if not parameter.requires_grad}
    missing = sorted(frozen - covered)
    unexpected = sorted(covered - frozen)
    report = {
        'rank': audit.rank,
        'frozen_parameters_unchanged': not changed and not missing and not unexpected,
        'changed_frozen_parameters': changed,
        'unchecked_frozen_parameters': missing,
        'unexpected_initial_frozen_parameters': unexpected,
        'frozen_tensors_checked': len(covered),
        'expert_weight_slices_checked': expert_slices
    }
    (audit.root / f'post_training_integrity_rank{audit.rank}.json').write_text(json.dumps(report, indent=2))
    assert report['frozen_parameters_unchanged'], report
    return report
