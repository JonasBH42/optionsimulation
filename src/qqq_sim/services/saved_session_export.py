"""Export stored session bytes and JSON without replaying historical semantics."""
from __future__ import annotations

from pathlib import Path
import sqlite3

from qqq_sim.broker.models import BrokerError
from qqq_sim.config import writable_path
from qqq_sim.serialization import digest, save_json
from qqq_sim.storage.events import EventStore, SCHEMA_VERSION

_TABLES = ("sessions", "branches", "commands", "events", "cash_postings", "checkpoints",
           "analyses", "analysis_requests", "projections")
_REQUIRED = {"sessions", "branches", "events", "checkpoints", "projections"}


def export_saved_session(source, destination, branch_id=None):
    """Read schema-v1 storage in one SQLite snapshot; never construct a controller.

    Sources are project-local so even SQLite's WAL shared-memory bookkeeping
    cannot create a sidecar beyond the user's hard write boundary. A copied
    external saved database can be placed in the project before this operation.
    """
    source = writable_path(source)
    destination = writable_path(destination)
    if not source.is_file():
        raise BrokerError("SAVED_SESSION_NOT_FOUND", path=str(source))
    protected = (source, Path(str(source) + "-wal"), Path(str(source) + "-shm"))
    outputs = (destination, destination.with_suffix(destination.suffix + ".tmp"))
    if any(path == original or (path.exists() and original.exists() and path.samefile(original))
           for path in outputs for original in protected):
        raise BrokerError("SAVED_SESSION_EXPORT_WOULD_OVERWRITE_SOURCE")
    db = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, isolation_level=None, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute("BEGIN")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise BrokerError("STORAGE_SCHEMA_UNSUPPORTED", schema_version=version)
        schema = [dict(row) for row in db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name")]
        tables = {row["name"] for row in schema}
        if not _REQUIRED <= tables or tables - set(_TABLES) - {"sqlite_sequence"}:
            raise BrokerError("SAVED_SESSION_SCHEMA_UNSUPPORTED")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise BrokerError("SAVED_SESSION_STORAGE_INTEGRITY_FAILURE")
        rows = [dict(row) for row in db.execute("SELECT * FROM sessions ORDER BY id")]
        if len(rows) != 1:
            raise BrokerError("SESSION_IDENTITY_AMBIGUOUS")
        session = rows[0]
        if session["schema_version"] != version:
            raise BrokerError("STORAGE_SCHEMA_UNSUPPORTED", schema_version=session["schema_version"])
        branches = {row["id"]: dict(row) for row in db.execute("SELECT * FROM branches ORDER BY id")}
        selected = branch_id or session["active_branch"]
        if selected not in branches:
            raise BrokerError("BRANCH_NOT_FOUND", branch_id=selected)
        for key, branch in branches.items():
            visited = set()
            cursor = key
            while cursor is not None:
                if cursor in visited or cursor not in branches:
                    raise BrokerError("SAVED_SESSION_BRANCH_GRAPH_INVALID")
                visited.add(cursor)
                cursor = branches[cursor]["parent_id"]
            if branch["session_id"] != session["id"]:
                raise BrokerError("SESSION_IDENTITY_AMBIGUOUS")
        # Bypass the writable constructor. These existing methods only SELECT
        # and validate JSON/hash chains; none applies pricing or account rules.
        reader = object.__new__(EventStore)
        reader.path, reader.connection = source, db
        restored = {key: reader.restore(key) for key in sorted(branches)}
        if "analysis_requests" in tables:
            reader.analysis_records(session_id=session["id"])
        raw = {table: [dict(row) for row in db.execute(f'SELECT * FROM "{table}" ORDER BY rowid')]
               for table in _TABLES if table in tables}
        state = restored[selected]["state"]
        record = {"export_schema": "QQQ_SAVED_SESSION_RAW_V1", "storage_schema_version": version,
                  "mode": "STORED_RECORDS_WITHOUT_REPLAY", "economic_reinterpretation": False,
                  "migration_applied": False, "source_path": str(source),
                  "session_id": session["id"], "branch_id": selected,
                  "state_version": restored[selected]["version"], "saved_state": state,
                  "implementation_identity": state.get("implementation_identity"),
                  "restored_stored_branches": restored, "stored_schema": schema, "stored_tables": raw,
                  "integrity": {"sqlite": "OK", "event_chains_and_checkpoints": "VERIFIED",
                                "modern_analysis_hashes": "VERIFIED" if "analysis_requests" in tables else "NOT_PRESENT"}}
        record["export_hash"] = digest(record)
        db.execute("COMMIT")
    except sqlite3.DatabaseError as exc:
        raise BrokerError("SAVED_SESSION_STORAGE_INVALID", detail=str(exc)) from exc
    finally:
        db.close()
    return save_json(destination, record)
