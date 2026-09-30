# -*- coding: utf-8 -*-
"""
unpack_configs.py
 
Constructs jitclass versions of every boolean-combination dict defined in
experiment_config, in one call, instead of unpacking them one by one.
 
Usage:
    import experiment_config, misc
    cfg = unpack_configs.unpack(experiment_config, misc)
 
    # then access by the original names:
    cfg.transition_path, cfg.experiment_building_rest, ...
"""

from types import SimpleNamespace
 
 
def unpack(config_module, misc):
    """Return a SimpleNamespace mapping each dict name in config_module
    to misc.construct_jitclass(that_dict)."""
    out = {}
    for name in dir(config_module):
        if name.startswith("_"):
            continue
        value = getattr(config_module, name)
        if isinstance(value, dict):
            out[name] = misc.construct_jitclass(value)
    return SimpleNamespace(**out)
 