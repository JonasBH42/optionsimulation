"""Pin the exact sources compiled into a standalone executable."""
from pathlib import Path
import hashlib
import json

ROOT=Path(r"D:\optionsimulation")


def current():
    sources={str(p.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((ROOT/"src").rglob("*.py")) if p.name!="_build_info.py"}
    for relative in ("packaging/main.py", "packaging/build.ps1", "packaging/pysidedeploy.spec", "pyproject.toml", "tools/build_source_manifest.py"):
        sources[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    lock=hashlib.sha256((ROOT/"requirements.lock").read_bytes()).hexdigest()
    build_id=hashlib.sha256(json.dumps({"sources":sources,"lock":lock},sort_keys=True).encode()).hexdigest()
    return build_id,sources,lock


def online_audit_sources(sources):
    """Embed original bytes as UTF-8 text; universal newline conversion is forbidden."""
    embedded = {}
    for path in sorted((ROOT/"src/qqq_sim/risk").glob("*.py")):
        key = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sources.get(key):
            raise RuntimeError("Source changed while preparing the build manifest: " + key)
        embedded[key] = raw.decode("utf-8")
    return embedded


if __name__=="__main__":
    build_id,sources,lock=current()
    audit_sources=online_audit_sources(sources)
    target=ROOT/"src/qqq_sim/_build_info.py"
    target.write_text('"""Exact sources and environment selected for the standalone build."""\n'
                      +f'BUILD_ID = {build_id!r}\nSOURCE_HASHES = {sources!r}\nLOCK_HASH = {lock!r}\nONLINE_AUDIT_SOURCES = {audit_sources!r}\n',encoding="utf-8")
    record={"build_id":build_id,"source_hashes":sources,"lock_hash":lock}
    (ROOT/"runtime/results/build-source-manifest.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(build_id)
