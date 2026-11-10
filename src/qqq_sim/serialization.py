"""Safe deterministic JSON: no pickle, eval, NaN or implicit infinity."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from collections.abc import Mapping
import hashlib
import json
import math


def primitive(value):
    if is_dataclass(value) and not isinstance(value, type):
        return primitive(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [primitive(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((primitive(v) for v in value), key=str)
    if isinstance(value, (Decimal, Path)):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return {"value": None, "reason": "NONFINITE", "sign": 1 if value > 0 else -1 if value < 0 else None}
    if hasattr(value, "tolist"):
        return primitive(value.tolist())
    return value


def dumps(value, *, pretty=False):
    return json.dumps(primitive(value), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, indent=2 if pretty else None,
                      separators=None if pretty else (",", ":"))


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def save_json(path, value):
    from .config import writable_path
    target = writable_path(path, create_parent=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(dumps(value, pretty=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
