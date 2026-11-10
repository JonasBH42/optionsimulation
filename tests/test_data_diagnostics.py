import csv
import gzip
from datetime import date, datetime
import shutil

import pytest

from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.fixtures import synthetic_mappings, write_synthetic_dataset
from qqq_sim.data.models import DataError
from qqq_sim.replay.calendar import NY


def make_data(tmp_path):
    manifest = write_synthetic_dataset(tmp_path / "data", minutes=3)
    return HistoricalData(manifest["underlying_root"], manifest["options_root"], synthetic_mappings())


def test_duplicate_oi_diagnostics_identify_every_conflicting_source(tmp_path):
    data = make_data(tmp_path)
    source = next((data.catalog.options_root / "oi_before").glob("*.csv.gz"))
    copies = [source.with_name(prefix + source.name) for prefix in ("copy_", "extra_")]
    for target in copies:
        shutil.copyfile(source, target)
    expected_paths = {str(path.resolve()) for path in (source, *copies)}

    with pytest.raises(DataError, match="DUPLICATE_LOGICAL_PARTITION") as caught:
        data.catalog.get("oi_before", date(2026, 11, 9), date(2026, 11, 13))
    assert set(caught.value.details["paths"]) == expected_paths
    assert caught.value.details["path"] in expected_paths

    snapshot = data.snapshot(datetime(2026, 11, 9, 9, 31, tzinfo=NY))
    warning = next(item for item in snapshot.diagnostics if item.code == "DUPLICATE_LOGICAL_PARTITION")
    assert warning.path in expected_paths and warning.severity == "WARNING"
    assert all(path in warning.message for path in expected_paths)
    assert snapshot.quotes and all(entry.known_oi is None for entry in snapshot.quotes)

    for target in copies:
        target.unlink()
    data.catalog.scan()
    assert data.catalog.get("oi_before", date(2026, 11, 9), date(2026, 11, 13)).path == str(source.resolve())
    assert not data.catalog._duplicate_paths


def test_volume_expiry_warning_preserves_source_row_and_rejects_volume(tmp_path):
    data = make_data(tmp_path)
    source = next((data.catalog.options_root / "ohlcv").glob("*.csv.gz"))
    with gzip.open(source, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        names, rows = reader.fieldnames, list(reader)
    rows[0]["expiration"] = "2026-11-20"
    with gzip.open(source, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)

    snapshot = data.snapshot(datetime(2026, 11, 9, 9, 31, tzinfo=NY))
    warning = next(item for item in snapshot.diagnostics if item.code == "EXPIRY_IDENTITY_MISMATCH")
    assert warning.path == str(source.resolve()) and warning.row_number == 2
    assert warning.severity == "WARNING" and "2026-11-20" in warning.message
    assert snapshot.quotes and all(entry.volume is None for entry in snapshot.quotes)
