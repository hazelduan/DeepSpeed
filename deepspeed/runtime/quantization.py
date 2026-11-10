# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Resolve training module selections and pass their recipes to TorchAO."""

from fnmatch import fnmatchcase

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.utils import logger


def _select_modules(model, format_name, config):
    patterns = config["module_name_patterns"]
    min_numel = config["min_numel"]
    linears, experts = [], []
    for name, module in model.named_modules():
        if not any(fnmatchcase(name, pattern) for pattern in patterns):
            continue
        if type(module) is torch.nn.Linear and module.weight.numel() >= min_numel:
            linears.append(name)
        elif isinstance(module, GroupedExperts):
            if min(module.w1[0].numel(), module.w2[0].numel(), module.w3[0].numel()) < min_numel:
                continue
            if getattr(module, "gate_up_impl", "auto") == "fused":
                raise RuntimeError(f"{format_name} experts require separate gate/up projections")
            experts.append(name)

    names = linears + experts
    if not names:
        raise RuntimeError(f"{format_name} module selection matched no nn.Linear or AutoEP GroupedExperts modules")
    expected = config["expected_module_count"]
    if expected is not None and len(names) != expected:
        raise RuntimeError(f"{format_name} expected_module_count={expected}, selected {len(names)} modules")
    unmatched = [pattern for pattern in patterns if not any(fnmatchcase(name, pattern) for name in names)]
    if unmatched:
        raise RuntimeError(f"{format_name} patterns matched no eligible modules: {unmatched}")
    return tuple(linears), tuple(experts)


def _hifloat8_recipe(config_class, options, linear, grouped):
    # The official AO config defaults to native ranges. Preserve the prior
    # DeepSpeed recipe for omitted fields, including in existing JSON files.
    policy = {"input_dst_type_max": 15, "weight_dst_type_max": 15, "grad_dst_type_max": 224}
    policy.update(options or {})
    return config_class.from_dict({"linear": policy if linear else None, "grouped": policy if grouped else None})


_RECIPES = {"hifloat8": _hifloat8_recipe}


def configure_training_quantization(model, configs, device):
    """Convert the selected modules in one pass after AutoEP and device placement."""
    accelerator = get_accelerator()
    selections, module_configs = {}, {}
    for format_name, config in configs.items():
        linears, experts = _select_modules(model, format_name, config)
        try:
            config_class = accelerator.get_training_quantization_config(format_name)
            if config_class is None or torch.device(device).type != accelerator.device_name():
                raise RuntimeError(f"No {format_name} training backend for {device}")
            recipe = _RECIPES[format_name](config_class, config["config"], bool(linears), bool(experts))
        except (ImportError, RuntimeError, ValueError, TypeError) as error:
            raise RuntimeError(f"{format_name} selected {len(linears)} Linear and {len(experts)} AutoEP modules, "
                               f"but its backend or recipe failed before conversion: {error}") from error

        leaves = [f"{name}.grouped_mm" if name else "grouped_mm" for name in experts]
        for name in (*linears, *leaves):
            if name in module_configs:
                raise RuntimeError(f"Training quantization formats overlap at {name!r}")
            module_configs[name] = recipe
        selections[format_name] = linears, experts

    from torchao.quantization import FqnToConfig, quantize_

    quantize_(model, FqnToConfig(module_configs), filter_fn=None)
    for format_name, (linears, experts) in selections.items():
        logger.info("Training quantization %s converted %d Linear and %d AutoEP modules", format_name, len(linears),
                    len(experts))
    return selections
