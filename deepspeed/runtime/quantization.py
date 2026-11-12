# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Resolve training module selections and pass their recipes to TorchAO."""

from fnmatch import fnmatchcase
import re

import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.moe.ep_experts import GroupedExperts
from deepspeed.utils import logger


def _select_modules(model, format_name, config):
    patterns = config["module_name_patterns"]
    min_numel = config["min_numel"]
    candidates = {}
    for name, module in model.named_modules():
        if patterns and not any(fnmatchcase(name, pattern) for pattern in patterns):
            continue
        if type(module) is torch.nn.Linear and module.weight.numel() >= min_numel:
            candidates[name] = module
        elif isinstance(module, GroupedExperts):
            if min(module.w1[0].numel(), module.w2[0].numel(), module.w3[0].numel()) >= min_numel:
                candidates[name] = module

    if not candidates:
        raise RuntimeError(f"{format_name} module selection matched no nn.Linear or AutoEP GroupedExperts modules")
    unmatched = [pattern for pattern in patterns if not any(fnmatchcase(name, pattern) for name in candidates)]
    if unmatched:
        raise RuntimeError(f"{format_name} patterns matched no eligible modules: {unmatched}")

    overrides = config.get("module_configs") or {}
    regexes = {key: re.compile(key[3:]) for key in overrides if key.startswith("re:")}
    for key in overrides:
        if key != "_default" and key not in candidates:
            if key not in regexes or not any(regexes[key].fullmatch(name) for name in candidates):
                raise RuntimeError(f"{format_name} module_configs selector matched no eligible modules: {key!r}")

    groups = {}
    for name, module in candidates.items():
        key = _module_recipe_key(name, overrides, regexes)
        override = overrides.get(key, {} if patterns else None)
        if override is None:
            continue
        policy = dict(config.get("config") or {})
        policy.update(override)
        _, linears, experts = groups.setdefault(key, (policy, [], []))
        if type(module) is torch.nn.Linear:
            linears.append(name)
        else:
            if module.gate_up_impl == "fused":
                raise RuntimeError(f"{format_name} experts require separate gate/up projections")
            experts.append(name)

    count = sum(len(linears) + len(experts) for _, linears, experts in groups.values())
    if not count:
        raise RuntimeError(f"{format_name} module selection matched no nn.Linear or AutoEP GroupedExperts modules")
    expected = config["expected_module_count"]
    if expected is not None and count != expected:
        raise RuntimeError(f"{format_name} expected_module_count={expected}, selected {count} modules")
    return groups.values()


def _module_recipe_key(name, overrides, regexes):
    if name in overrides:
        return name
    for key, regex in regexes.items():
        if regex.fullmatch(name):
            return key
    return "_default"


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
        selected_linears, selected_experts = [], []
        for options, linears, experts in _select_modules(model, format_name, config):
            try:
                config_class = accelerator.get_training_quantization_config(format_name)
                if config_class is None or torch.device(device).type != accelerator.device_name():
                    raise RuntimeError(f"No {format_name} training backend for {device}")
                recipe = _RECIPES[format_name](config_class, options, bool(linears), bool(experts))
            except (ImportError, RuntimeError, ValueError, TypeError) as error:
                raise RuntimeError(f"{format_name} selected {len(linears)} Linear and {len(experts)} AutoEP modules, "
                                   f"but its backend or recipe failed before conversion: {error}") from error

            leaves = [f"{name}.grouped_mm" if name else "grouped_mm" for name in experts]
            for name in (*linears, *leaves):
                if name in module_configs:
                    raise RuntimeError(f"Training quantization formats overlap at {name!r}")
                module_configs[name] = recipe
            selected_linears.extend(linears)
            selected_experts.extend(experts)
        selections[format_name] = tuple(selected_linears), tuple(selected_experts)

    from torchao.quantization import FqnToConfig, quantize_

    quantize_(model, FqnToConfig(module_configs), filter_fn=None)
    for format_name, (linears, experts) in selections.items():
        logger.info("Training quantization %s converted %d Linear and %d AutoEP modules", format_name, len(linears),
                    len(experts))
    return selections
