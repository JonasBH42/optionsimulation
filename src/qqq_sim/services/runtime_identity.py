"""Content identity for session semantics; no environment paths enter compatibility."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sqlite3
import struct
import sys

from qqq_sim import __version__
from qqq_sim.broker.models import BrokerError
from qqq_sim.serialization import digest

IDENTITY_SCHEMA = "QQQ_SESSION_IMPLEMENTATION_V1"
RUNTIME_DEPENDENCIES = ("PySide6", "pyqtgraph", "numpy", "scipy", "pyarrow", "duckdb", "tzdata")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_identity(package_root, embedded):
    """Use loaded package sources, never a different checkout beside an executable."""
    package_root = Path(package_root)
    if (package_root / "services/runtime_identity.py").is_file():
        sources = {"src/qqq_sim/" + path.relative_to(package_root).as_posix(): _sha256(path.read_bytes())
                   for path in sorted(package_root.rglob("*.py")) if path.name != "_build_info.py"}
        lock_path = package_root.parent.parent / "requirements.lock"
        if "src/qqq_sim/__init__.py" not in sources or not lock_path.is_file():
            raise BrokerError("IMPLEMENTATION_IDENTITY_UNAVAILABLE", reason="Live package or dependency lock is incomplete")
        return sources, _sha256(lock_path.read_bytes()), {"source_origin": "LIVE_SOURCE"}

    sources = getattr(embedded, "SOURCE_HASHES", None)
    lock_hash = getattr(embedded, "LOCK_HASH", None)
    build_id = getattr(embedded, "BUILD_ID", None)
    if (not isinstance(sources, dict) or not sources or not isinstance(lock_hash, str)
            or not _SHA256.fullmatch(lock_hash)
            or any(not isinstance(key, str) or not isinstance(value, str) or not _SHA256.fullmatch(value)
                   for key, value in sources.items())):
        raise BrokerError("IMPLEMENTATION_BUILD_MANIFEST_INVALID")
    # This is the exact encoding used by tools/build_source_manifest.py. It
    # verifies the embedded manifest's integrity, not publisher authenticity.
    expected = _sha256(json.dumps({"sources": sources, "lock": lock_hash}, sort_keys=True).encode())
    if build_id != expected:
        raise BrokerError("IMPLEMENTATION_BUILD_MANIFEST_INVALID")
    for key, source in getattr(embedded, "ONLINE_AUDIT_SOURCES", {}).items():
        if not isinstance(source, str) or _sha256(source.encode("utf-8")) != sources.get(key):
            raise BrokerError("IMPLEMENTATION_BUILD_MANIFEST_INVALID", source=key)
    selected = {key: value for key, value in sources.items()
                if key.startswith("src/qqq_sim/") and key.endswith(".py") and not key.endswith("/_build_info.py")}
    required = {"src/qqq_sim/__init__.py", "src/qqq_sim/services/runtime_identity.py"}
    if not required <= selected.keys() or any(".." in Path(key).parts or "\\" in key for key in selected):
        raise BrokerError("IMPLEMENTATION_BUILD_MANIFEST_INVALID")
    return selected, lock_hash, {"source_origin": "EMBEDDED_BUILD", "embedded_build_id": build_id}


def _make_identity(sources, lock_hash, dependencies, python_runtime, provenance):
    content = {"application": "qqq-simulator", "application_version": __version__,
               "source_hashes": sources, "source_digest": digest(sources),
               "dependency_lock_hash": lock_hash, "runtime_dependencies": dependencies,
               "python_runtime": python_runtime}
    return {"schema": IDENTITY_SCHEMA, "identity_hash": digest({"schema": IDENTITY_SCHEMA, "content": content}),
            "content": content, "provenance": provenance}


@lru_cache(maxsize=1)
def _cached_identity():
    from qqq_sim import _build_info
    sources, lock_hash, provenance = _source_identity(Path(__file__).resolve().parent.parent, _build_info)
    try:
        dependencies = {name: metadata.version(name) for name in RUNTIME_DEPENDENCIES}
    except metadata.PackageNotFoundError as exc:
        raise BrokerError("IMPLEMENTATION_DEPENDENCY_IDENTITY_UNAVAILABLE", dependency=str(exc)) from exc
    python_runtime = {"implementation": platform.python_implementation(), "version": platform.python_version(),
                      "word_size": struct.calcsize("P") * 8, "byte_order": sys.byteorder,
                      "sqlite_library_version": sqlite3.sqlite_version}
    return _make_identity(sources, lock_hash, dependencies, python_runtime, provenance)


def runtime_identity():
    """Hash source/dependency content once per process; callers get detached records."""
    return deepcopy(_cached_identity())


def validate_runtime_identity(saved):
    """Validate a stored record without authorizing its implementation to run."""
    if saved is None:
        raise BrokerError("SESSION_IMPLEMENTATION_IDENTITY_MISSING", remedy="Export the original saved session without replay")
    if (not isinstance(saved, dict) or saved.get("schema") != IDENTITY_SCHEMA
            or not isinstance(saved.get("content"), dict)
            or saved.get("identity_hash") != digest({"schema": IDENTITY_SCHEMA, "content": saved.get("content")})):
        raise BrokerError("SESSION_IMPLEMENTATION_IDENTITY_INVALID")
    return deepcopy(saved)


def verify_runtime_identity(saved):
    """Refuse old or incompatible semantics before reconstructing account objects."""
    saved = validate_runtime_identity(saved)
    current = runtime_identity()
    if saved["identity_hash"] != current["identity_hash"]:
        raise BrokerError("SESSION_IMPLEMENTATION_REVISION_CHANGED", saved=saved["identity_hash"],
                          current=current["identity_hash"], remedy="Use the original build or export the saved session without replay")
    return deepcopy(saved)
