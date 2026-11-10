"""Standalone builds retain identities and inspect pinned online source text."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from qqq_sim import _build_info
from qqq_sim.risk.contracts import RiskError
from qqq_sim.risk.offline import price_basis, pricing_oracle, validate


def _compiled_audit(monkeypatch, tmp_path):
    root = Path(validate.__file__).parent.parent
    sources = {"src/qqq_sim/risk/" + name: (root/name).read_bytes().decode("utf-8")
               for name in validate.ONLINE_AUDIT_FILES}
    hashes = {key: hashlib.sha256(source.encode("utf-8")).hexdigest() for key, source in sources.items()}
    monkeypatch.setattr(_build_info, "SOURCE_HASHES", hashes)
    monkeypatch.setattr(_build_info, "ONLINE_AUDIT_SOURCES", sources)
    monkeypatch.setattr(validate, "__file__", str(tmp_path/"compiled/qqq_sim/risk/offline/validate.py"))
    return sources, hashes


def test_compiled_price_basis_resumes_same_source_identity_without_function_code(monkeypatch, tmp_path):
    source_result = price_basis.build_price_basis(tmp_path/"basis", budget_seconds=1, max_initial_cells=0)
    checkpoint = Path(source_result["checkpoint"])
    identity = json.loads(checkpoint.read_text())["identity"]
    hashes = {"src/qqq_sim/risk/offline/" + name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
              for name, module in (("price_basis.py", price_basis), ("pricing_oracle.py", pricing_oracle))}
    monkeypatch.setattr(_build_info, "SOURCE_HASHES", hashes)
    monkeypatch.setattr(price_basis, "__file__", str(tmp_path/"compiled/price_basis.py"))
    monkeypatch.setattr(pricing_oracle, "__file__", str(tmp_path/"compiled/pricing_oracle.py"))
    # Native compiled functions need not expose CPython's __code__ attribute.
    monkeypatch.setattr(price_basis, "american_price", object())
    oracle = price_basis.OraclePriceCache(tmp_path/"cache")
    assert oracle.kernel_hash == hashes["src/qqq_sim/risk/offline/pricing_oracle.py"]
    resumed = price_basis.build_price_basis(tmp_path/"basis", budget_seconds=1, max_initial_cells=0)
    assert resumed["status"] == "COMPLETE"
    assert json.loads(checkpoint.read_text())["identity"] == identity


@pytest.mark.parametrize("digest", [None, "unknown", "z" * 64])
def test_compiled_kernel_missing_or_invalid_digest_refuses(monkeypatch, tmp_path, digest):
    key = "src/qqq_sim/risk/offline/pricing_oracle.py"
    monkeypatch.setattr(_build_info, "SOURCE_HASHES", {key: digest})
    monkeypatch.setattr(pricing_oracle, "__file__", str(tmp_path/"absent.py"))
    with pytest.raises(RiskError) as exc:
        price_basis.OraclePriceCache(tmp_path/"cache")
    assert exc.value.code == "IMPLEMENTATION_IDENTITY_UNAVAILABLE"


def test_compiled_online_audit_inspects_identical_pinned_sources(monkeypatch, tmp_path):
    source_result = validate.online_import_audit()
    _compiled_audit(monkeypatch, tmp_path)
    compiled_result = validate.online_import_audit()
    assert compiled_result["passed"] == source_result["passed"]
    assert compiled_result["violations"] == source_result["violations"]
    assert [item["sha256"] for item in compiled_result["audited_sources"]] == [item["sha256"] for item in source_result["audited_sources"]]
    assert all(item["origin"] == "VERIFIED_BUILD_SOURCE" for item in compiled_result["audited_sources"])


def test_compiled_online_audit_detects_forbidden_import_in_verified_embedded_text(monkeypatch, tmp_path):
    sources, hashes = _compiled_audit(monkeypatch, tmp_path)
    key = "src/qqq_sim/risk/hold.py"
    sources[key] += "\nfrom .offline import pricing_oracle\n"
    hashes[key] = hashlib.sha256(sources[key].encode("utf-8")).hexdigest()
    result = validate.online_import_audit()
    assert not result["passed"]
    assert any(item["file"] == "hold.py" and item["module"] == "offline" for item in result["violations"])


@pytest.mark.parametrize("mutation,expected", [("missing", "ONLINE_AUDIT_SOURCE_UNAVAILABLE"), ("tampered", "ONLINE_AUDIT_SOURCE_INTEGRITY_FAILURE")])
def test_compiled_online_audit_never_assumes_missing_or_tampered_evidence_passed(monkeypatch, tmp_path, mutation, expected):
    sources, _ = _compiled_audit(monkeypatch, tmp_path)
    key = "src/qqq_sim/risk/hold.py"
    if mutation == "missing":
        sources.pop(key)
    else:
        sources[key] += "\n# changed after manifest capture\n"
    with pytest.raises(RiskError) as exc:
        validate.online_import_audit()
    assert exc.value.code == expected


def test_build_manifest_embeds_exact_newlines_and_rejects_source_race(monkeypatch, tmp_path):
    path = Path(__file__).parents[1]/"tools/build_source_manifest.py"
    spec = importlib.util.spec_from_file_location("test_build_source_manifest", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    source = tmp_path/"src/qqq_sim/risk/hold.py"
    source.parent.mkdir(parents=True)
    raw = b"# exact Windows source\r\nVALUE = 1\r\n"
    source.write_bytes(raw)
    hashes = {"src/qqq_sim/risk/hold.py": hashlib.sha256(raw).hexdigest()}
    assert builder.online_audit_sources(hashes)["src/qqq_sim/risk/hold.py"].encode("utf-8") == raw
    source.write_bytes(raw + b"# changed\r\n")
    with pytest.raises(RuntimeError, match="Source changed"):
        builder.online_audit_sources(hashes)
