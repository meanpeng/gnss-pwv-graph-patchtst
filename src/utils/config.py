"""
Configuration loader with CLI override support.
"""
import yaml
import os
from typing import Dict, Any


class Config:
    """Simple nested dict wrapper for attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default=None):
        return self._config.get(key, default)
    
    def to_dict(self):
        return self._config
    
    def __repr__(self):
        return f"Config({self._config})"


def load_config(path: str) -> Config:
    """Load YAML config file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def override_config(config: Config, overrides: Dict[str, Any]) -> Config:
    """Override config values (flat key format like 'training.lr')."""
    config_dict = config.to_dict()
    
    for key, value in overrides.items():
        keys = key.split(".")
        target = config_dict
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value
    
    return Config(config_dict)
