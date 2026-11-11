"""Explicit, copied compatibility upgrade for the reviewed replay predecessor."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from qqq_sim.broker.models import BrokerError, canonical, fingerprint
from qqq_sim.config import AppConfig, writable_path
from qqq_sim.serialization import digest
from qqq_sim.storage.events import EventStore, SCHEMA_VERSION
from ._session_upgrade_policy import POLICY_ID, PREDECESSOR_IDENTITIES
from .runtime_identity import runtime_identity, validate_runtime_identity, verify_runtime_identity


_TABLES = {"sessions", "branches", "commands", "events", "cash_postings", "checkpoints",
           "analyses", "analysis_requests", "projections", "sqlite_sequence"}
_REQUIRED = {"sessions", "branches", "events", "checkpoints", "projections"}


def is_upgradeable_identity(saved):
    """Recognize the two reviewed predecessors, never arbitrary old code."""
    saved = validate_runtime_identity(saved)
    current = runtime_identity()
    content, now = saved["content"], current["content"]
    return (any(saved["identity_hash"] == predecessor["identity_hash"]
                and content == predecessor["content"] for predecessor in PREDECESSOR_IDENTITIES)
            and content.get("source_digest") == digest(content.get("source_hashes"))
            and all(content.get(key) == now.get(key) for key in
                    ("application", "application_version", "dependency_lock_hash",
                     "runtime_dependencies", "python_runtime")))


def _require_predecessor(saved):
    if not is_upgradeable_identity(saved):
        # Keep the existing precise missing/invalid/revision errors.
        verify_runtime_identity(saved)
        raise BrokerError("SESSION_UPGRADE_NOT_REQUIRED")


@contextmanager
def _verified_reader(path, branch_id=None):
    """Validate stored evidence in one read-only SQLite snapshot."""
    path = writable_path(path)
    if not path.is_file():
        raise BrokerError("SAVED_SESSION_NOT_FOUND", path=str(path))
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute("BEGIN")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise BrokerError("STORAGE_SCHEMA_UNSUPPORTED", schema_version=version)
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not _REQUIRED <= tables or tables - _TABLES:
            raise BrokerError("SAVED_SESSION_SCHEMA_UNSUPPORTED")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise BrokerError("SAVED_SESSION_STORAGE_INTEGRITY_FAILURE")
        sessions = [dict(row) for row in db.execute("SELECT * FROM sessions")]
        if len(sessions) != 1:
            raise BrokerError("SESSION_IDENTITY_AMBIGUOUS")
        session = sessions[0]
        if session["schema_version"] != version:
            raise BrokerError("STORAGE_SCHEMA_UNSUPPORTED", schema_version=session["schema_version"])
        branches = {row["id"]: dict(row) for row in db.execute("SELECT * FROM branches")}
        selected = branch_id or session["active_branch"]
        if selected not in branches:
            raise BrokerError("BRANCH_NOT_FOUND", branch_id=selected)
        for key, branch in branches.items():
            visited, cursor = set(), key
            while cursor is not None:
                if cursor in visited or cursor not in branches:
                    raise BrokerError("SAVED_SESSION_BRANCH_GRAPH_INVALID")
                visited.add(cursor)
                cursor = branches[cursor]["parent_id"]
            if branch["session_id"] != session["id"]:
                raise BrokerError("SESSION_IDENTITY_AMBIGUOUS")
        reader = object.__new__(EventStore)
        reader.path, reader.connection = path, db
        recovered = {key: reader.restore(key) for key in branches}
        if "analysis_requests" in tables:
            reader.analysis_records(session_id=session["id"])
        yield reader, session, recovered[selected]
    except sqlite3.DatabaseError as exc:
        raise BrokerError("SAVED_SESSION_STORAGE_INVALID", detail=str(exc)) from exc
    finally:
        db.close()


def upgrade_legacy_branch(store, branch_id, sequence=None, reason="Replay performance upgrade"):
    """Atomically fork a known legacy state; never rewrite its history or account.

    The caller must retain strict identity checks when loading the returned state.
    No broker objects or historical pricing code are needed for this transition.
    """
    parent = store.restore(branch_id, through_sequence=sequence)
    old_state = parent["state"]
    _require_predecessor(old_state.get("implementation_identity"))
    state = deepcopy(old_state)
    current = runtime_identity()
    state["implementation_identity"] = current
    new_branch, command_id = str(uuid4()), "upgrade:" + uuid4().hex
    state_record = {"version": parent["version"] + 1, "state": state}
    transition = {"policy_id": POLICY_ID, "reason": reason,
                  "previous_identity": old_state["implementation_identity"], "current_identity": current,
                  "parent_branch": parent["branch_id"], "parent_sequence": parent["sequence"],
                  "parent_state_hash": parent["state_hash"], "parent_event_hash": parent["event_hash"],
                  "economic_state_changed": False}
    stamp = old_state.get("decision_at")
    if not stamp or datetime.fromisoformat(stamp).tzinfo is None:
        raise BrokerError("SESSION_INITIALIZATION_INCOMPLETE")
    config = state.get("config")
    if config is None:
        row = store.connection.execute("SELECT config_json FROM sessions WHERE id=?", (parent["session_id"],)).fetchone()
        config = json.loads(row[0])
    db = store.connection
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute("INSERT INTO branches VALUES(?,?,?,?,?,?,?)",
                   (new_branch, parent["session_id"], branch_id, parent["sequence"], parent["event_hash"],
                    fingerprint(config), reason))
        previous_hash, final_sequence = parent["event_hash"], parent["sequence"]
        for source_id, event_type, payload, phase, utc_ns in (
                (command_id, "IMPLEMENTATION_UPGRADED", transition, "IMPLEMENTATION",
                 int(datetime.fromisoformat(stamp).timestamp() * 1_000_000_000)),
                ("commit:" + command_id, "STATE_COMMITTED", state_record, "COMMIT", 0)):
            final_sequence += 1
            event_id = new_branch + ":" + source_id
            hashed = {"branch_id": new_branch, "sequence": final_sequence, "event_id": event_id,
                      "utc_ns": utc_ns, "phase": phase, "event_type": event_type,
                      "schema_version": SCHEMA_VERSION, "payload": payload, "previous_hash": previous_hash}
            event_hash = fingerprint(hashed)
            db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (new_branch, final_sequence, event_id, utc_ns, phase, event_type, SCHEMA_VERSION,
                        canonical(payload), previous_hash, event_hash))
            previous_hash = event_hash
        db.execute("INSERT INTO projections VALUES(?,?,?,?,?)",
                   ("state", new_branch, final_sequence, SCHEMA_VERSION, canonical(state_record)))
        store._checkpoint_row(new_branch, parent["sequence"],
                              {"version": parent["version"], "state": old_state})
        store._checkpoint_row(new_branch, final_sequence, state_record)
        response = {"version": state_record["version"], "sequence": final_sequence,
                    "result": {"status": "UPGRADED", "policy_id": POLICY_ID}, "state_hash": fingerprint(state)}
        db.execute("INSERT INTO commands VALUES(?,?,?,?,?,?)",
                   (new_branch, command_id, parent["version"], canonical({"action": "UPGRADE_IMPLEMENTATION", **transition}),
                    "COMPLETED", canonical(response)))
        db.execute("UPDATE sessions SET active_branch=? WHERE id=?", (new_branch, parent["session_id"]))
        db.execute("COMMIT")
    except Exception:
        db.execute("ROLLBACK")
        raise
    return store.restore(new_branch)


def _verify_reusable_copy(destination, source):
    with _verified_reader(destination) as (reader, session, recovered):
        verify_runtime_identity(recovered["state"].get("implementation_identity"))
        rows = reader.connection.execute("SELECT branch_id,payload_json FROM events WHERE event_type='IMPLEMENTATION_UPGRADED'")
        for row in rows:
            value = json.loads(row["payload_json"])
            if (value.get("policy_id") == POLICY_ID
                    and value.get("parent_branch") == source["branch_id"]
                    and value.get("parent_sequence") == source["sequence"]
                    and value.get("parent_state_hash") == source["state_hash"]
                    and value.get("parent_event_hash") == source["event_hash"]
                    and value.get("current_identity", {}).get("identity_hash") == runtime_identity()["identity_hash"]
                    and session["id"] == source["session_id"]):
                branch = reader.connection.execute("SELECT parent_id,fork_sequence,fork_state_hash FROM branches WHERE id=?",
                                                   (row["branch_id"],)).fetchone()
                parent = reader.restore(source["branch_id"], through_sequence=source["sequence"])
                if (tuple(branch) == (source["branch_id"], source["sequence"], source["event_hash"])
                        and parent["state"] == source["state"] and parent["event_hash"] == source["event_hash"]
                        and value.get("previous_identity") == source["state"]["implementation_identity"]):
                    return destination, recovered["branch_id"]
        raise BrokerError("SESSION_UPGRADE_COPY_ORIGIN_MISMATCH", path=str(destination))


def prepare_session_restore(path, branch_id=None):
    """Return an exact-current save or a verified reusable upgraded working copy."""
    path = writable_path(path)
    with _verified_reader(path, branch_id) as (reader, session, source):
        saved = validate_runtime_identity(source["state"].get("implementation_identity"))
        if saved["identity_hash"] == runtime_identity()["identity_hash"]:
            return path, branch_id
        _require_predecessor(saved)
        config = AppConfig(**source["state"].get("config", json.loads(session["config_json"]))).validate()
        key = digest({"source": str(path), "branch": source["branch_id"], "event_hash": source["event_hash"],
                      "state_hash": source["state_hash"], "target_identity": runtime_identity()["identity_hash"]})
        destination = writable_path(Path(config.runtime_root) / "session_upgrades" / key / "session.sqlite", create_parent=True)
        if destination.exists():
            return _verify_reusable_copy(destination, source)
        stage = destination.with_name(".stage-" + uuid4().hex + ".sqlite")
        try:
            target = sqlite3.connect(str(stage), isolation_level=None)
            try:
                reader.connection.backup(target)
            finally:
                target.close()
            store = EventStore(stage)
            try:
                # A single staged file must contain its entire durable state at publication.
                store.connection.execute("PRAGMA journal_mode=DELETE")
                upgrade_legacy_branch(store, source["branch_id"])
            finally:
                store.connection.close()
            _verify_reusable_copy(stage, source)
            with stage.open("rb+") as handle:
                os.fsync(handle.fileno())
            try:
                # Atomic exclusive publication: never replace an already progressing copy.
                os.link(stage, destination)
            except FileExistsError:
                pass
            return _verify_reusable_copy(destination, source)
        finally:
            for temporary in (stage, Path(str(stage) + "-wal"), Path(str(stage) + "-shm")):
                temporary.unlink(missing_ok=True)
