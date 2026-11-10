"""Bounded schema/coverage audits; metadata-only scans remain cheap by design."""
from __future__ import annotations
from collections import Counter
from datetime import date
from pathlib import Path
from decimal import Decimal, InvalidOperation
import csv
import gzip
from .catalog import file_hash
from .models import DataError, json_value, stable_hash
from .normalization import canonical_schema, row_timestamp, parse_timestamp, field, normalize_right


def audit_session(data, day: date, expiry: date | None = None, *, hash_sources: bool = True) -> dict:
    """Inspect one requested day without concatenating or repairing an archive."""
    import pyarrow.parquet as pq
    if data.catalog.report is None:
        data.catalog.scan()
    session = data.calendar.session(day)
    result = {"schema_version": 1, "day": day.isoformat(), "calendar": json_value(session),
              "status": "OK", "sources": [], "diagnostics": [], "coverage": {}}
    chosen = [item for item in data.catalog.report.partitions if item.key.session_date == day and (expiry is None or item.key.expiration in (None, expiry))]
    for item in chosen:
        entry = {"key": json_value(item.key), "path": item.path, "size": item.size,
                 "mtime_ns": item.mtime_ns, "sha256": file_hash(item.path) if hash_sources else None}
        try:
            earliest = latest = None
            timestamp_errors, contracts = 0, set()
            contract_count = None
            if item.key in data.catalog._duplicates:
                raise DataError("DUPLICATE_LOGICAL_PARTITION", str(item.key))
            if item.key.data_type == "underlying":
                metadata = pq.read_metadata(item.path)
                schema = pq.read_schema(item.path)
                entry.update(schema=str(schema), raw_row_count=metadata.num_rows)
                names = tuple(schema.names)
                stamp_column = data.mapping("underlying").timestamp_column or next((name for name in ("regular_time", "datetime", "timestamp", "time", "date") if name in names), None)
                if stamp_column:
                    for batch in pq.ParquetFile(item.path).iter_batches(columns=[stamp_column], batch_size=4096):
                        for raw in batch.column(0).to_pylist():
                            try:
                                stamp = parse_timestamp(raw, day)
                                earliest = stamp if earliest is None else min(earliest, stamp)
                                latest = stamp if latest is None else max(latest, stamp)
                            except ValueError:
                                timestamp_errors += 1
                rows, boundary, diagnostics = data.underlying_reader.read(item, session.option_close, data.mapping("underlying"))
                labels = set(row.label for row in rows)
                expected = data.calendar.minute_labels(day)
                result["coverage"] = {"expected_slots": len(expected), "valid_rth_rows": len(rows),
                                      "boundary_rows": len(boundary), "missing_labels": [value.isoformat() for value in expected if value not in labels],
                                      "first_valid": rows[0].label.isoformat() if rows else None,
                                      "last_valid": rows[-1].label.isoformat() if rows else None}
                result["diagnostics"].extend(json_value(diagnostics))
            else:
                names, records = data.theta_reader.iter_records(item)
                aliases, count, contracts_capped = canonical_schema(names), 0, False
                for number, values in records:
                    count += 1
                    row = dict(zip(names, values))
                    if aliases.get("timestamp") or aliases.get("ms_of_day"):
                        try:
                            stamp = row_timestamp(row, aliases, day, data.mapping(item.key.data_type))
                            earliest = stamp if earliest is None else min(earliest, stamp)
                            latest = stamp if latest is None else max(latest, stamp)
                        except ValueError:
                            timestamp_errors += 1
                    if not contracts_capped:
                        try:
                            strike = Decimal(str(field(row, "strike", aliases)))
                            right = normalize_right(field(row, "right", aliases))
                            if not strike.is_finite() or strike <= 0:
                                raise ValueError()
                            contracts.add((row.get("root", row.get("symbol", "QQQ")), right, strike))
                            if len(contracts) > 65_536:
                                contracts_capped = True
                                contracts.clear()
                        except (ValueError, InvalidOperation):
                            pass
                contract_count = None if contracts_capped else len(contracts)
                entry.update(schema=list(names), aliases=aliases, raw_row_count=count,
                             iv_unit=data.mapping(item.key.data_type).iv_unit,
                             strike_divisor=str(data.mapping(item.key.data_type).strike_divisor) if data.mapping(item.key.data_type).strike_divisor else None)
                if contracts_capped:
                    result["diagnostics"].append({"code": "CONTRACT_COUNT_BUDGET_REACHED", "message": "More than65536 distinct source contract keys; count remains unknown", "path": item.path})
            if timestamp_errors:
                result["diagnostics"].append({"code": "INVALID_RAW_TIMESTAMPS", "message": str(timestamp_errors), "path": item.path})
            if hash_sources and file_hash(item.path) != entry["sha256"]:
                raise DataError("DATA_REVISION_CHANGED", "Source changed during bounded audit")
            quality = "METADATA_AUDITED_WITH_DIAGNOSTICS" if timestamp_errors else "METADATA_AUDITED"
            entry.update(contract_count=contract_count, earliest_raw_at=earliest.isoformat() if earliest else None,
                         latest_raw_at=latest.isoformat() if latest else None, quality_status=quality,
                         normalization_version=data.mapping(item.key.data_type).mapping_version)
            data.catalog.record_audit(item.key, sha256=entry["sha256"], schema=tuple(names), row_count=entry["raw_row_count"],
                                      schema_fingerprint=stable_hash(entry["schema"]), contract_count=contract_count,
                                      earliest_raw_at=earliest, latest_raw_at=latest, quality_status=quality,
                                      normalization_version=entry["normalization_version"])
        except (DataError, OSError) as exc:
            result["diagnostics"].append({"code": getattr(exc, "code", "READ_FAILED"), "message": str(exc), "path": item.path})
        result["sources"].append(entry)
    if not chosen:
        result["diagnostics"].append({"code": "ARCHIVE_SESSION_MISSING", "message": day.isoformat()})
    if result["diagnostics"] or result["coverage"].get("missing_labels"):
        result["status"] = "PARTIAL"
    result["diagnostic_counts"] = dict(Counter(item["code"] for item in result["diagnostics"]))
    result["scope"] = "One selected session; no repair, acquisition, synthetic fallback, or historical rule certification."
    if data.catalog.manifest_path:
        data.catalog.persist()
    return result
