"""Versioned local configuration and the user's hard output boundary."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os

PROJECT_ROOT = Path(r"D:\optionsimulation").resolve()


class ConfigurationError(ValueError):
    pass


def writable_path(value: str | Path, *, create_parent: bool = False) -> Path:
    """Reject traversal, external output paths, and junction/symlink escapes."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ConfigurationError(f"OUTPUT_OUTSIDE_PROJECT: {path}")
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def configure_environment() -> None:
    """Keep dependency, build, Qt, and test caches in the project."""
    locations = {
        "TEMP": ".tmp", "TMP": ".tmp", "TMPDIR": ".tmp",
        "PIP_CACHE_DIR": ".cache/pip", "UV_CACHE_DIR": ".cache/uv",
        "UV_PYTHON_INSTALL_DIR": ".runtime/python",
        "TORCH_HOME": ".cache/torch", "TORCHINDUCTOR_CACHE_DIR": ".cache/torchinductor",
        "XDG_CACHE_HOME": ".cache", "XDG_CONFIG_HOME": "runtime/config",
        "XDG_DATA_HOME": "runtime/data", "MPLCONFIGDIR": ".cache/matplotlib",
        "NUITKA_CACHE_DIR": ".cache/nuitka",
    }
    for variable, relative in locations.items():
        path = writable_path(relative)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")


@dataclass(frozen=True)
class AppConfig:
    schema_version: int = 1
    underlying_root: str = r"D:\QQQ\OHLCV_daily"
    options_root: str = r"D:\QQQ\raw_theta"
    runtime_root: str = "runtime"
    pack_path: str | None = None
    normalization_path: str | None = None
    parameters_path: str | None = None
    calendar_path: str | None = None
    initial_cash: str = "5000.00"
    bp_cap: str = "5000.00"
    loss_cap: str = "1000.00"
    session_loss_cap: str = "1000.00"
    rule_profile: str = "TT_COMPONENTS_20261028_SIM_V1"
    account_profile: str = "OPTIONS_COLLATERAL_QQQ_V1"
    rule_mode: str = "fixed_rules_counterfactual"
    rule_asof: str = "2026-10-28"
    operational_model: str = "SIMULTANEOUS_DELIVERY_HEDGE_V1"
    calendar_profile: str = "STANDARD_US_CALENDAR_SCENARIO_V1"
    strict: bool = True
    io_workers: int = 2
    offline_workers: int = 6
    cache_bytes: int = 512 * 1024 * 1024

    def validate(self) -> "AppConfig":
        from decimal import Decimal
        if self.schema_version != 1:
            raise ConfigurationError("CONFIG_SCHEMA_UNSUPPORTED")
        writable_path(self.runtime_root)
        if self.rule_mode not in ("fixed_rules_counterfactual", "strict_historical"):
            raise ConfigurationError("INVALID_RULE_MODE")
        if self.rule_profile != "TT_COMPONENTS_20261028_SIM_V1" or self.rule_asof != "2026-10-28":
            raise ConfigurationError("UNSUPPORTED_RULE_PROFILE: supply a supported complete versioned profile")
        if self.account_profile != "OPTIONS_COLLATERAL_QQQ_V1":
            raise ConfigurationError("UNSUPPORTED_ACCOUNT_PROFILE")
        if self.calendar_profile not in ("STANDARD_US_CALENDAR_SCENARIO_V1","DETERMINISTIC_US_EQUITY_SCENARIO_V1") and self.calendar_path is None:
            raise ConfigurationError("CALENDAR_RECORDS_REQUIRED")
        if not isinstance(self.strict,bool):
            raise ConfigurationError("STRICT_FLAG_MUST_BE_BOOLEAN")
        for field in (self.initial_cash, self.bp_cap, self.loss_cap, self.session_loss_cap):
            if not isinstance(field,(str,Decimal,int)) or isinstance(field,bool):
                raise ConfigurationError("MONEY_REQUIRES_EXACT_DECIMAL_INPUT")
            number = Decimal(field)
            if not number.is_finite() or number <= 0 or number != number.quantize(Decimal('.01')):
                raise ConfigurationError("INVALID_ACCOUNT_LIMIT")
        if any(isinstance(v,bool) or not isinstance(v,int) for v in (self.io_workers,self.offline_workers,self.cache_bytes)):
            raise ConfigurationError("RESOURCE_LIMIT_MUST_BE_INTEGER")
        if not (1 <= self.io_workers <= 4 and 1 <= self.offline_workers <= 6):
            raise ConfigurationError("RESOURCE_LIMIT_EXCEEDED")
        if not 1_048_576 <= self.cache_bytes <= 4*1024**3:
            raise ConfigurationError("CACHE_BUDGET_OUTSIDE_SUPPORTED_RANGE")
        for source in (self.underlying_root, self.options_root):
            resolved = Path(source).resolve()
            if writable_path(self.runtime_root).is_relative_to(resolved):
                raise ConfigurationError("SOURCE_OUTPUT_OVERLAP")
        return self

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        if path is None:
            return cls().validate()
        values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ConfigurationError(f"UNKNOWN_CONFIG_FIELDS: {sorted(unknown)}")
        return cls(**values).validate()

    def save(self, path: str | Path) -> Path:
        target = writable_path(path, create_parent=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return target
