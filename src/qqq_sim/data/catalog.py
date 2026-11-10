"""Metadata-first archive indexing and immutable run revisions."""
from __future__ import annotations
from dataclasses import replace
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
import os
import json
import time
import uuid
from .models import CatalogReport, DataError, Diagnostic, Partition, PartitionKey, stable_hash

UNDERLYING_NAME = re.compile(r"QQQ_1m_(\d{4}-\d{2}-\d{2})\.parquet\Z")
SESSION_NAME = re.compile(r"(?<!\d)(\d{8})(?!\d)")
EXPIRY_NAME = re.compile(r"exp=(\d{4}-\d{2}-\d{2})")
DATA_TYPES = ("greeks", "ohlcv", "trades", "oi_before", "oi")
REPLACE_RETRY_DELAYS = (.01, .02, .04, .08, .16, .25, .25)


def _replace_catalog(temporary, target):
    """Windows scanners/readers may briefly deny an otherwise atomic rename."""
    for attempt in range(len(REPLACE_RETRY_DELAYS)+1):
        try:
            os.replace(temporary,target)
            return
        except OSError as exc:
            # ERROR_ACCESS_DENIED can be permanent; bounded retries preserve
            # that original failure while tolerating transient sharing locks.
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt==len(REPLACE_RETRY_DELAYS):
                raise
            time.sleep(REPLACE_RETRY_DELAYS[attempt])


def file_hash(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CatalogService:
    def __init__(self, underlying_root: str | Path, options_root: str | Path, manifest_path=None, normalization_identity=None):
        self.underlying_root, self.options_root = Path(underlying_root), Path(options_root)
        self.report: CatalogReport | None = None
        self._index: dict[PartitionKey, Partition] = {}
        self._duplicates: set[PartitionKey] = set()
        self._duplicate_paths: dict[PartitionKey, tuple[str, ...]] = {}
        self._bound: dict[str, tuple[int, int, str]] = {}
        self.manifest_path = manifest_path
        self._directories = {}
        self._last_directory_check = 0.
        self.directory_check_interval = .5
        self._directory_revision_error = None
        self.normalization_identity = normalization_identity
        self.source_identity = None

    def scan(self, roots=None, previous_manifest=None) -> CatalogReport:
        if roots:
            self.underlying_root, self.options_root = map(Path, roots)
        self.source_identity = stable_hash({"roots": [str(self.underlying_root.resolve()), str(self.options_root.resolve())], "normalization": self.normalization_identity})
        if previous_manifest is None and self.report is None and self.manifest_path and Path(self.manifest_path).is_file():
            try:
                previous_manifest = json.loads(Path(self.manifest_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous_manifest = None
        if previous_manifest and previous_manifest.get("source_identity") != self.source_identity:
            previous_manifest = None
        previous = {item.path: item for item in self.report.partitions} if self.report and self.report.source_identity == self.source_identity else {}
        for raw in (previous_manifest or {}).get("partitions", ()):
            try:
                item = dict(raw)
                key = item.pop("key")
                key = PartitionKey(key["data_type"], date.fromisoformat(key["session_date"]), date.fromisoformat(key["expiration"]) if key.get("expiration") else None)
                for name in ("earliest_raw_at", "latest_raw_at", "audited_at"):
                    if item.get(name):
                        item[name] = datetime.fromisoformat(item[name])
                item["schema"] = tuple(item.get("schema", ()))
                restored = Partition(key, **item)
                previous[restored.path] = restored
            except (KeyError, TypeError, ValueError):
                continue
        partitions, diagnostics, seen, duplicates, duplicate_paths, directories = [], [], {}, set(), {}, {}
        def remember_directory(path):
            if path.is_dir():
                directories[str(path)] = path.stat().st_mtime_ns
        def files(root, suffix):
            for folder, children, names in os.walk(root):
                children.sort()
                remember_directory(Path(folder))
                for name in sorted(names):
                    if name.endswith(suffix):
                        yield Path(folder) / name
        for root in (self.underlying_root, self.options_root):
            ancestor = root
            while not ancestor.is_dir() and ancestor != ancestor.parent:
                ancestor = ancestor.parent
            remember_directory(ancestor)
            remember_directory(root.parent)
        remember_directory(self.options_root)
        def append(path: Path, key: PartitionKey):
            stat = path.stat()
            item = Partition(key, str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            old = previous.get(item.path)
            if old is not None and old.key == key and (old.size, old.mtime_ns) == (item.size, item.mtime_ns):
                item = old
            if key in seen:
                duplicates.add(key)
                duplicate_paths.setdefault(key, [seen[key].path]).append(item.path)
                diagnostics.append(Diagnostic("DUPLICATE_LOGICAL_PARTITION", f"{key}: {seen[key].path} and {path}", str(path)))
            seen[key] = item
            partitions.append(item)
        if not self.underlying_root.is_dir():
            diagnostics.append(Diagnostic("UNDERLYING_ROOT_MISSING", "Authoritative underlying archive is missing", str(self.underlying_root)))
        else:
            for path in files(self.underlying_root, ".parquet"):
                match = UNDERLYING_NAME.fullmatch(path.name)
                if not match:
                    continue
                try:
                    append(path, PartitionKey("underlying", date.fromisoformat(match[1])))
                except ValueError:
                    diagnostics.append(Diagnostic("INVALID_FILENAME_DATE", path.name, str(path)))
        for kind in DATA_TYPES:
            folder = self.options_root / kind
            if not folder.is_dir():
                diagnostics.append(Diagnostic("OPTION_FOLDER_MISSING", f"Missing {kind}", str(folder), severity="WARNING" if kind != "greeks" else "ERROR"))
                continue
            for path in files(folder, ".csv.gz"):
                days, expiries = SESSION_NAME.findall(path.name), EXPIRY_NAME.findall(path.name)
                if len(days) != 1 or len(expiries) != 1:
                    diagnostics.append(Diagnostic("INVALID_PARTITION_NAME", path.name, str(path)))
                    continue
                try:
                    append(path, PartitionKey(kind, datetime.strptime(days[0], "%Y%m%d").date(), date.fromisoformat(expiries[0])))
                except ValueError:
                    diagnostics.append(Diagnostic("INVALID_FILENAME_DATE", path.name, str(path)))
        self.report = CatalogReport(tuple(partitions), tuple(diagnostics), (str(self.underlying_root), str(self.options_root)), source_identity=self.source_identity)
        self._index, self._duplicates = seen, duplicates
        self._duplicate_paths = {key: tuple(paths) for key, paths in duplicate_paths.items()}
        self._directories = directories
        self._last_directory_check = time.monotonic()
        if self.manifest_path:
            self.persist()
        return self.report

    def refresh_if_changed(self, force=False):
        if self._directory_revision_error:
            raise DataError("DATA_REVISION_CHANGED", "A source partition was added to a bound session", added_paths=self._directory_revision_error)
        if self.report is None:
            return self.scan()
        now = time.monotonic()
        if not force and now-self._last_directory_check < self.directory_check_interval:
            return self.report
        self._last_directory_check = now
        changed = False
        for path, previous in self._directories.items():
            try:
                changed = Path(path).stat().st_mtime_ns != previous
            except OSError:
                changed = True
            if changed:
                break
        if changed:
            before = {item.path for item in self.report.partitions}
            bound_days = {item.key.session_date for item in self.report.partitions if item.path in self._bound}
            self.verify_revision()
            self.scan()
            added = [item.path for item in self.report.partitions if item.path not in before and item.key.session_date in bound_days]
            if added:
                self._directory_revision_error = tuple(added)
                raise DataError("DATA_REVISION_CHANGED", "A source partition was added to a bound session", added_paths=added)
        return self.report

    def record_audit(self, key, **observed):
        item = self.get(key.data_type, key.session_date, key.expiration)
        updated = replace(item, **observed, audited_at=datetime.now(timezone.utc))
        self._index[key] = updated
        self.report = replace(self.report, partitions=tuple(updated if prior.path == item.path else prior for prior in self.report.partitions))
        return updated

    def persist(self, destination=None):
        from qqq_sim.config import writable_path
        target = writable_path(destination or self.manifest_path, create_parent=True)
        temporary = target.with_name(target.name+"."+uuid.uuid4().hex+".part")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(self.report.to_dict(), stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            _replace_catalog(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def get(self, kind: str, day: date, expiry: date | None = None) -> Partition:
        self.refresh_if_changed()
        key = PartitionKey(kind, day, expiry)
        if key in self._duplicates:
            paths = self._duplicate_paths[key]
            raise DataError("DUPLICATE_LOGICAL_PARTITION", f"{key}: conflicting sources {'; '.join(paths)}",
                            path=paths[0], paths=paths)
        if key not in self._index:
            raise DataError("PARTITION_MISSING", str(key))
        item = self._index[key]
        try:
            stat = Path(item.path).stat()
        except OSError as exc:
            raise DataError("SOURCE_FILE_UNAVAILABLE", str(exc), path=item.path) from exc
        if item.path in self._bound:
            size, mtime, fingerprint = self._bound[item.path]
            if stat.st_size != size or stat.st_mtime_ns != mtime:
                raise DataError("DATA_REVISION_CHANGED", "Source changed during a bound run", path=item.path)
        if stat.st_size != item.size or stat.st_mtime_ns != item.mtime_ns:
            item = Partition(item.key, item.path, stat.st_size, stat.st_mtime_ns)
            self._index[key] = item
            self.report = replace(self.report, partitions=tuple(item if prior.path == item.path else prior for prior in self.report.partitions))
        return item

    def expirations(self, day: date, kind: str = "greeks") -> tuple[date, ...]:
        self.refresh_if_changed()
        return tuple(sorted({key.expiration for key in self._index if key.data_type == kind and key.session_date == day and key.expiration is not None}))

    def bind_revision(self, session_config=None, keys=None) -> dict:
        """Hash only explicitly bound partitions; metadata scans never decompress all data."""
        if self.report is None:
            self.scan()
        selected = list(keys) if keys is not None else list(self._index)
        if self._duplicates.intersection(selected):
            raise DataError("DUPLICATE_LOGICAL_PARTITION", "Resolve duplicate sources before binding")
        records = []
        for key in sorted(selected):
            item = self.get(key.data_type, key.session_date, key.expiration)
            digest = file_hash(item.path)
            self._bound[item.path] = item.size, item.mtime_ns, digest
            records.append({"path": item.path, "size": item.size, "mtime_ns": item.mtime_ns, "sha256": digest})
        return {"revision_id": stable_hash(records), "sources": records}

    def verify_revision(self, deep: bool = False) -> None:
        if self._directory_revision_error:
            raise DataError("DATA_REVISION_CHANGED", "A source partition was added to a bound session", added_paths=self._directory_revision_error)
        for path, (size, mtime, digest) in self._bound.items():
            try:
                stat = Path(path).stat()
            except OSError as exc:
                raise DataError("SOURCE_FILE_UNAVAILABLE", str(exc), path=path) from exc
            if stat.st_size != size or stat.st_mtime_ns != mtime or (deep and file_hash(path) != digest):
                raise DataError("DATA_REVISION_CHANGED", "Bound source no longer matches", path=path)
