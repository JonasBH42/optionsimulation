from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
import json
import shutil
import pytest
from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.audit import audit_session
from qqq_sim.data.fixtures import write_synthetic_dataset, synthetic_mappings
from qqq_sim.data.models import DataError, NormalizationMapping
from qqq_sim.data.normalization import normalize_vendor_greeks
from qqq_sim.replay.calendar import NY


def dataset(tmp_path, name="data", catalog_path=None, mappings=None):
    manifest = write_synthetic_dataset(tmp_path / name, minutes=4)
    return HistoricalData(manifest["underlying_root"], manifest["options_root"], mappings or synthetic_mappings(), catalog_path=catalog_path)


def test_added_same_day_partition_is_discovered_without_manual_rescan(tmp_path):
    data = dataset(tmp_path)
    empty = data.catalog.options_root / "greeks" / "initially_empty"
    empty.mkdir()
    report = data.catalog.scan()
    data.catalog.directory_check_interval = 0
    source = next(item for item in report.partitions if item.key.data_type == "greeks")
    new_expiry = date(2026, 11, 20)
    target = empty / "new_20261109_exp=2026-11-20.csv.gz"
    shutil.copyfile(source.path, target)
    assert new_expiry in data.catalog.expirations(date(2026, 11, 9))
    assert data.catalog.get("greeks", date(2026, 11, 9), new_expiry).path == str(target.resolve())


def test_new_partition_in_bound_day_is_a_persistent_revision_failure(tmp_path):
    data = dataset(tmp_path)
    report = data.catalog.scan()
    data.catalog.directory_check_interval = 0
    source = next(item for item in report.partitions if item.key.data_type == "greeks")
    data.catalog.bind_revision(keys=[source.key])
    target = data.catalog.options_root / "greeks" / "new_20261109_exp=2026-11-20.csv.gz"
    shutil.copyfile(source.path, target)
    with pytest.raises(DataError, match="DATA_REVISION_CHANGED"):
        data.catalog.expirations(date(2026, 11, 9))
    target.unlink()
    with pytest.raises(DataError, match="DATA_REVISION_CHANGED"):
        data.catalog.verify_revision()


def test_bounded_audit_metadata_persists_and_only_matching_roots_and_mapping_reuse_it(tmp_path):
    manifest_path = tmp_path / "metadata.json"
    data = dataset(tmp_path, catalog_path=manifest_path)
    result = audit_session(data, date(2026, 11, 9))
    assert result["sources"]
    known = data.catalog.get("greeks", date(2026, 11, 9), date(2026, 11, 13))
    assert known.row_count == 40 and known.contract_count == 10
    assert known.sha256 and known.schema_fingerprint and known.normalization_version
    assert known.earliest_raw_at < known.latest_raw_at and known.audited_at
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["source_identity"] == data.catalog.source_identity
    reopened = HistoricalData(data.catalog.underlying_root, data.catalog.options_root, synthetic_mappings(), catalog_path=manifest_path)
    assert reopened.catalog.get("greeks", date(2026, 11, 9), date(2026, 11, 13)).row_count == 40
    different_mapping = {"options": replace(synthetic_mappings()["options"], mapping_version="OTHER_MAPPING")}
    changed = HistoricalData(data.catalog.underlying_root, data.catalog.options_root, different_mapping, catalog_path=manifest_path)
    assert changed.catalog.get("greeks", date(2026, 11, 9), date(2026, 11, 13)).row_count is None
    other = dataset(tmp_path, "another_data", catalog_path=manifest_path)
    assert other.catalog.get("greeks", date(2026, 11, 9), date(2026, 11, 13)).row_count is None


def test_greek_conversions_require_exact_known_units_and_preserve_vendor_values(tmp_path):
    mapping = replace(synthetic_mappings()["options"], greek_units=(("delta", "decimal"), ("gamma", "per_dollar"), ("vega", "per_unit"), ("theta", "per_day")))
    data = dataset(tmp_path, mappings={"options": mapping})
    quote = data.snapshot(datetime(2026, 11, 9, 9, 31, tzinfo=NY)).quotes[0].latest_event
    assert dict(quote.vendor_greeks)["vega"] == .10
    assert dict(quote.normalized_greeks)["vega"] == pytest.approx(.001)
    assert dict(quote.normalized_greek_units)["vega"] == "per_1pct"
    converted, units, flags = normalize_vendor_greeks((("delta", 50.), ("gamma", 2.), ("theta", -365.), ("charm", .2)),
        NormalizationMapping(greek_units=(("delta", "percent"), ("gamma", "per_dollar_per_contract"), ("theta", "per_year_act365"))), 100)
    assert dict(converted) == {"delta": .5, "gamma": .02, "theta": -1.}
    assert flags == ("VENDOR_GREEK_UNITS_UNVERIFIED:charm",)
    plain = dataset(tmp_path, "plain")
    unknown = plain.snapshot(datetime(2026, 11, 9, 9, 31, tzinfo=NY)).quotes[0].latest_event
    assert not unknown.normalized_greeks and "VENDOR_GREEK_UNITS_UNVERIFIED:vega" in unknown.quality_flags


def test_mirror_identity_is_persisted_and_mapping_change_invalidates_cache(tmp_path):
    import pyarrow as pa
    from qqq_sim.data.catalog import file_hash
    from qqq_sim.data.mirror import write_mirror, valid_mirror
    data = dataset(tmp_path, catalog_path=tmp_path / "catalog.json")
    partition = data.catalog.get("underlying", date(2026, 11, 9))
    table = pa.table({"available_at": [1], "contract_key": ["QQQ"], "session": ["2026-11-09"], "source_hash": [file_hash(partition.path)]})
    target = write_mirror(partition.path, table, tmp_path / "normalized.parquet", mapping_identity="EXPLICIT_MAPPING_A", catalog=data.catalog, partition_key=partition.key)
    assert data.catalog.get("underlying", date(2026, 11, 9)).mirror_identity
    assert valid_mirror(partition.path, target, mapping_identity="EXPLICIT_MAPPING_A")
    assert not valid_mirror(partition.path, target, mapping_identity="EXPLICIT_MAPPING_B")


@pytest.mark.parametrize('winerror',[5,32,33])
def test_catalog_atomic_publish_retries_transient_windows_locks(tmp_path,monkeypatch,winerror):
    import qqq_sim.data.catalog as module
    path=tmp_path/'metadata.json'; data=dataset(tmp_path,catalog_path=path)
    data.catalog.scan()
    previous=path.read_bytes()
    data.catalog.record_audit(data.catalog.report.partitions[0].key,row_count=123)
    replacement=module.os.replace; calls=[]; delays=[]
    def locked_then_available(source,target):
        assert target.read_bytes()==previous
        calls.append(source.read_bytes())
        if len(calls)<=3:
            failure=PermissionError('injected Windows sharing/access lock'); failure.winerror=winerror
            raise failure
        return replacement(source,target)
    monkeypatch.setattr(module.os,'replace',locked_then_available)
    monkeypatch.setattr(module.time,'sleep',delays.append)
    data.catalog.persist()
    assert len(calls)==4 and len(set(calls))==1
    assert delays==list(module.REPLACE_RETRY_DELAYS[:3])
    assert path.read_bytes()==calls[0] and path.read_bytes()!=previous
    assert not list(tmp_path.glob('metadata.json.*.part'))


@pytest.mark.parametrize('winerror,expected_attempts',[(5,8),(32,8),(87,1),(112,1)])
def test_catalog_publish_preserves_old_bytes_and_surfaces_permanent_error(tmp_path,monkeypatch,winerror,expected_attempts):
    import qqq_sim.data.catalog as module
    path=tmp_path/'metadata.json'; data=dataset(tmp_path,catalog_path=path)
    data.catalog.scan()
    previous=path.read_bytes(); calls=[]; delays=[]
    failure=PermissionError('injected permanent publish failure'); failure.winerror=winerror
    def permanent(source,target):
        calls.append(source)
        assert target.read_bytes()==previous
        raise failure
    monkeypatch.setattr(module.os,'replace',permanent)
    monkeypatch.setattr(module.time,'sleep',delays.append)
    with pytest.raises(PermissionError) as caught:
        data.catalog.persist()
    assert caught.value is failure and len(calls)==expected_attempts
    assert len(delays)==expected_attempts-1 and sum(delays)<1
    assert path.read_bytes()==previous
    assert not list(tmp_path.glob('metadata.json.*.part'))
