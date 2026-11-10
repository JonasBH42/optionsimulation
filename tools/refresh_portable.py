"""Copy only the fresh compiler output; preserve the previous distribution."""
from pathlib import Path
import json
import os
import shutil
import sys
from uuid import uuid4

ROOT=Path(r'D:\optionsimulation')
sys.path.insert(0,str(ROOT/'src'))
from qqq_sim.config import writable_path


def main():
    source=writable_path('packaging/deployment/main.dist')
    destination=writable_path('dist/QQQSimulator.dist')
    if not (source/'QQQSimulator.exe').is_file():raise RuntimeError('Successful standalone compiler output required')
    size=sum(path.stat().st_size for path in source.rglob('*') if path.is_file())
    if shutil.disk_usage(ROOT).free<size+64*1024**2:raise RuntimeError('Insufficient space for a fresh portable copy')
    archive=writable_path(Path('runtime/previous-distributions')/('QQQSimulator-'+uuid4().hex),create_parent=True)
    saved=False
    if destination.exists():
        os.replace(destination,archive)
        saved=True
    try:
        shutil.copytree(source,destination)
    except Exception:
        if destination.exists():
            failed=writable_path(str(archive)+'-incomplete',create_parent=True)
            os.replace(destination,failed)
        if saved:os.replace(archive,destination)
        raise
    print(json.dumps({'status':'FRESH_COMPILER_OUTPUT_COPIED','bytes':size,'destination':str(destination),
                      'previous_distribution':str(archive) if saved else None}))


if __name__=='__main__':main()
