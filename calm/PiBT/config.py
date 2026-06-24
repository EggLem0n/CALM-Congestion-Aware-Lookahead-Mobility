"""YAML config access.

The full parameter list lives in configs/default.yaml (single source of truth).
This module only loads it and exposes it as an attribute-style, read-only object.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

# This file is calm/mapf/config.py, so the repo root (which holds configs/)
# is three levels up.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class Config:
    """Attribute-style, read-only view over the YAML settings.

    ``config.max_time == values["max_time"]``. Accessing a key not present in
    configs/default.yaml raises immediately so typos surface fast.
    """

    def __init__(self, values: Dict[str, Any]):
        object.__setattr__(self, "_values", dict(values))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(
                f"Unknown config key '{name}'. Define it in configs/default.yaml first."
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Config is read-only. Use config.replace(key=value) instead.")

    def replace(self, **overrides: Any) -> "Config":
        """Return a copy with some values overridden (like dataclasses.replace)."""
        merged = dict(self._values)
        merged.update(overrides)
        return Config(merged)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._values)

    def __repr__(self) -> str:
        return f"Config({self._values!r})"


def load_config(path: Optional[str] = None) -> Config:
    """Load configs/default.yaml, optionally overlaying another YAML file.

    The overlay may only use keys that exist in default.yaml; unknown keys raise
    so typos do not silently do nothing.
    """
    import yaml

    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Required config registry not found: {DEFAULT_CONFIG_PATH}")
    values = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        overlay = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(overlay, dict):
            raise ValueError(f"Config file must contain a mapping of settings: {path}")
        unknown_keys = sorted(set(overlay) - set(values))
        if unknown_keys:
            raise ValueError(f"Unknown config keys in {path}: {unknown_keys}")
        values.update(overlay)
    return Config(values)


# Type annotations across modules say MAPFConfig.
MAPFConfig = Config
