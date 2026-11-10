"""Atomic, provenance-bound derived caches, always inside the project."""
from __future__ import annotations
import json
import os
import uuid
from pathlib import Path
from .catalog import file_hash
from .models import DataError, stable_hash
from qqq_sim.config import writable_path


def write_mirror(source_path, table, destination, mapping_version="QQQ_NORMALIZATION_V1", *, mapping_identity=None, catalog=None, partition_key=None):
    import pyarrow.parquet as pq
    target = writable_path(destination, create_parent=True)
    source = Path(source_path).resolve()
    if source == target:
        raise DataError("SOURCE_OUTPUT_OVERLAP", "Mirror cannot overwrite source")
    required = {"available_at", "contract_key", "session", "source_hash"}
    if not required.issubset(table.column_names):
        raise DataError("MIRROR_PROVENANCE_MISSING", "Canonical mirror requires availability, identity, session and source hash", missing=sorted(required - set(table.column_names)))
    fingerprint = file_hash(source)
    import pyarrow.compute as pc
    if table.num_rows and (pc.any(pc.is_null(table["source_hash"])).as_py() or not pc.all(pc.equal(table["source_hash"], fingerprint)).as_py()):
        raise DataError("MIRROR_SOURCE_HASH_MISMATCH", "Every canonical row must identify the original source bytes")
    metadata = dict(table.schema.metadata or {})
    identity = mapping_identity or mapping_version
    metadata.update({b"raw_sha256": fingerprint.encode(), b"mapping_version": mapping_version.encode(), b"mapping_identity": identity.encode(), b"complete": b"true"})
    table = table.replace_schema_metadata(metadata)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".part")
    pq.write_table(table, temporary)
    restored = pq.read_metadata(temporary)
    if restored.num_rows != table.num_rows or file_hash(source) != fingerprint:
        temporary.unlink(missing_ok=True)
        raise DataError("MIRROR_VALIDATION_FAILED", "Source changed or row count failed")
    os.replace(temporary, target)
    if catalog is not None and partition_key is not None:
        partition = catalog.get(partition_key.data_type, partition_key.session_date, partition_key.expiration)
        if Path(partition.path).resolve() != source:
            raise DataError("MIRROR_SOURCE_IDENTITY_MISMATCH", "Catalog key does not identify mirror source")
        catalog.record_audit(partition_key, mirror_identity=stable_hash({"path": str(target), "raw_sha256": fingerprint,
                                                                        "mapping_identity": identity, "mirror_sha256": file_hash(target)}))
        if catalog.manifest_path:
            catalog.persist()
    return target


def valid_mirror(source_path, destination, mapping_version="QQQ_NORMALIZATION_V1", *, mapping_identity=None) -> bool:
    import pyarrow.parquet as pq
    destination = Path(destination)
    if not destination.is_file():
        return False
    metadata = pq.read_metadata(destination).metadata or {}
    return metadata.get(b"raw_sha256") == file_hash(source_path).encode() and metadata.get(b"mapping_version") == mapping_version.encode() and metadata.get(b"mapping_identity") == (mapping_identity or mapping_version).encode() and metadata.get(b"complete") == b"true"
