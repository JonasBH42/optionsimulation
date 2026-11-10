from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from types import SimpleNamespace

import pytest

from qqq_sim.broker.models import BrokerError
from qqq_sim.cli import demo_config
from qqq_sim.config import AppConfig
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController
from qqq_sim.serialization import digest
from qqq_sim.services import runtime_identity as identity
from qqq_sim.services.saved_session_export import export_saved_session


def _embedded(sources, lock_hash, audit=None):
    build_id = hashlib.sha256(json.dumps({"sources": sources, "lock": lock_hash}, sort_keys=True).encode()).hexdigest()
    return SimpleNamespace(SOURCE_HASHES=sources, LOCK_HASH=lock_hash, BUILD_ID=build_id,
                           ONLINE_AUDIT_SOURCES=audit or {})


def test_live_and_embedded_content_identity_match_and_ignore_build_provenance(tmp_path):
    package = tmp_path / "src/qqq_sim"
    (package / "services").mkdir(parents=True)
    (package / "__init__.py").write_bytes(b'__version__ = "0.1.0"\r\n')
    (package / "services/runtime_identity.py").write_bytes(b"# exact source bytes\r\n")
    generated = package / "_build_info.py"
    generated.write_text("BUILD_ID = 'DEVELOPMENT'")
    (tmp_path / "requirements.lock").write_bytes(b"numpy==1.0\r\n")
    live = identity._source_identity(package, SimpleNamespace())
    generated.write_text("BUILD_ID = 'different generated data'")
    assert identity._source_identity(package, SimpleNamespace()) == live
    manifest = {**live[0], "packaging/main.py": "a" * 64, "tools/build_source_manifest.py": "b" * 64}
    embedded = identity._source_identity(tmp_path / "compiled", _embedded(manifest, live[1]))
    assert live[:2] == embedded[:2]
    a = identity._make_identity(*live[:2], {"numpy": "1.0"}, {"version": "3.12.16"}, live[2])
    b = identity._make_identity(*embedded[:2], {"numpy": "1.0"}, {"version": "3.12.16"}, embedded[2])
    assert a["identity_hash"] == b["identity_hash"] and a["provenance"] != b["provenance"]
    (package / "services/runtime_identity.py").write_bytes(b"# changed semantics\r\n")
    assert identity._source_identity(package, SimpleNamespace())[0] != live[0]


@pytest.mark.parametrize("damage", ["build", "audit", "missing_source", "lock"])
def test_embedded_manifest_is_verified(tmp_path, damage):
    source = "# embedded\r\n"
    sources = {"src/qqq_sim/__init__.py": "a" * 64,
               "src/qqq_sim/services/runtime_identity.py": "b" * 64,
               "src/qqq_sim/risk/hold.py": hashlib.sha256(source.encode()).hexdigest()}
    module = _embedded(sources, "c" * 64, {"src/qqq_sim/risk/hold.py": source})
    if damage == "build": module.BUILD_ID = "d" * 64
    elif damage == "audit": module.ONLINE_AUDIT_SOURCES["src/qqq_sim/risk/hold.py"] = "changed"
    elif damage == "missing_source":
        sources.pop("src/qqq_sim/services/runtime_identity.py")
        module = _embedded(sources, "c" * 64)
    else: module.LOCK_HASH = "DEVELOPMENT"
    with pytest.raises(BrokerError, match="IMPLEMENTATION_BUILD_MANIFEST_INVALID"):
        identity._source_identity(tmp_path, module)


def test_runtime_identity_cached_once_and_detached(monkeypatch):
    from functools import lru_cache
    calls = []
    record = identity.runtime_identity()
    def compute():
        calls.append(True)
        return deepcopy(record)
    monkeypatch.setattr(identity, "_cached_identity", lru_cache(maxsize=1)(compute))
    first = identity.runtime_identity()
    first["content"]["runtime_dependencies"]["numpy"] = "tampered"
    assert identity.runtime_identity() == record
    assert calls == [True]
    assert set(record["content"]["runtime_dependencies"]) == set(identity.RUNTIME_DEPENDENCIES)
    assert record["content"]["application_version"] == "0.1.0"
    assert len(record["content"]["dependency_lock_hash"]) == 64
    assert all("_build_info.py" not in name for name in record["content"]["source_hashes"])


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "fixture"
    write_synthetic_dataset(root, minutes=35)
    return replace(demo_config(AppConfig(), root), runtime_root=str(tmp_path / "runtime"))


def _controller(config):
    return ReplayController.create(config, local_datetime(datetime(2026, 11, 9, 10, 1)))


def test_new_session_restore_export_and_branch_preserve_frozen_identity(config):
    controller = _controller(config)
    try:
        before = controller.broker.to_dict()
        version = controller.version
        frozen = controller._state()["implementation_identity"]
        exported = controller.export_record()
        assert exported["implementation_identity"] == frozen
        assert exported["state"]["implementation_identity"] == frozen
        # Export records are detached and cannot alter the next committed basis.
        exported["implementation_identity"]["content"]["runtime_dependencies"].clear()
        assert controller._state()["implementation_identity"] == frozen
        assert controller.version == version and controller.broker.to_dict() == before
        session = controller.session_id
        controller.branch()
        assert controller.version == version and controller._state()["implementation_identity"] == frozen
        assert controller.broker.to_dict() == before
    finally:
        controller.close()
    restored = ReplayController.restore(config, session)
    try:
        assert restored.version == version and restored._state()["implementation_identity"] == frozen
        assert restored.broker.to_dict() == before
    finally:
        restored.close()


def _write_incompatible_state(controller, damage):
    state = controller._state()
    if damage == "missing":
        state.pop("implementation_identity")
    else:
        frozen = state["implementation_identity"]
        frozen["content"]["runtime_dependencies"]["numpy"] = "previous-runtime-version"
        if damage == "mismatch":
            frozen["identity_hash"] = digest({"schema": frozen["schema"], "content": frozen["content"]})
    controller.store.transact(controller.branch_id, "simulate-previous-implementation", controller.version,
                              {"action": "FIXTURE"}, [], {}, state)
    return state


@pytest.mark.parametrize(("damage", "reason"), [
    ("missing", "SESSION_IMPLEMENTATION_IDENTITY_MISSING"),
    ("tamper", "SESSION_IMPLEMENTATION_IDENTITY_INVALID"),
    ("mismatch", "SESSION_IMPLEMENTATION_REVISION_CHANGED"),
])
def test_restore_refuses_missing_tampered_or_different_semantics_before_account_rebuild(config, monkeypatch, damage, reason):
    controller = _controller(config)
    session = controller.session_id
    _write_incompatible_state(controller, damage)
    controller.close()
    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible saved state must be refused before rebuilding account objects")
    monkeypatch.setattr("qqq_sim.replay.controller.Broker.from_dict", forbidden)
    with pytest.raises(BrokerError, match=reason):
        ReplayController.restore(config, session)


@pytest.mark.parametrize("damage", ["missing", "mismatch"])
def test_raw_export_retains_legacy_or_other_build_records_without_replay_or_writes(config, tmp_path, monkeypatch, damage):
    controller = _controller(config)
    source = controller.store.path
    saved = _write_incompatible_state(controller, damage)
    controller.store.record_analysis("legacy-request", controller.branch_id, controller.version,
                                     "selection", "original-model", "UNAVAILABLE", {"reason": "ORIGINAL_REASON"})
    # A schema-v1 file predating request archival has no analysis_requests table.
    if damage == "missing": controller.store.connection.execute("DROP TABLE analysis_requests")
    controller.close()
    before = source.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("Raw export must not construct a writable store or reinterpret broker state")
    monkeypatch.setattr("qqq_sim.storage.events.EventStore.__init__", forbidden)
    monkeypatch.setattr("qqq_sim.replay.controller.Broker.from_dict", forbidden)
    monkeypatch.setattr("qqq_sim.services.runtime_identity.runtime_identity", forbidden)
    output = export_saved_session(source, tmp_path / "raw-export.json")
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["saved_state"] == saved
    assert result["implementation_identity"] == saved.get("implementation_identity")
    assert result["economic_reinterpretation"] is False and result["migration_applied"] is False
    assert result["stored_tables"]["analyses"][0]["payload_json_or_path"] == '{"reason":"ORIGINAL_REASON"}'
    assert result["stored_tables"]["events"][-1]["event_type"] == "STATE_COMMITTED"
    assert result["integrity"]["event_chains_and_checkpoints"] == "VERIFIED"
    expected = result.pop("export_hash")
    assert digest(result) == expected
    assert source.read_bytes() == before


def test_raw_export_refuses_corrupted_event_chain(config, tmp_path):
    controller = _controller(config)
    source = controller.store.path
    controller.store.connection.execute("UPDATE events SET event_hash=?", ("0" * 64,))
    controller.close()
    with pytest.raises(BrokerError, match="EVENT_HASH_MISMATCH"):
        export_saved_session(source, tmp_path / "must-not-exist.json")
    assert not (tmp_path / "must-not-exist.json").exists()


def test_raw_export_cli_reads_incompatible_session_by_id_without_changing_it(config, tmp_path, capsys):
    from qqq_sim.cli import main
    controller = _controller(config)
    session, source = controller.session_id, controller.store.path
    saved = _write_incompatible_state(controller, "missing")
    controller.close()
    config_path = config.save(tmp_path / "config.json")
    destination = tmp_path / "cli-raw-export.json"
    before = source.read_bytes()
    assert main(["--config", str(config_path), "export-saved", "--session", session,
                 "--output", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "READ_ONLY_SAVED_STATE_EXPORT"
    assert json.loads(destination.read_text())["saved_state"] == saved
    assert source.read_bytes() == before


def test_raw_export_rejects_overwriting_source_or_wal(config):
    controller = _controller(config)
    source = controller.store.path
    try:
        for destination in (source, str(source) + "-wal", str(source) + "-shm"):
            with pytest.raises(BrokerError, match="SAVED_SESSION_EXPORT_WOULD_OVERWRITE_SOURCE"):
                export_saved_session(source, destination)
    finally:
        controller.close()
