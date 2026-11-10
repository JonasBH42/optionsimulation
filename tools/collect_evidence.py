"""Reproducible environment/source/test/license evidence. Writes only project files."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import csv
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(r"D:\optionsimulation")
sys.path.insert(0,str(ROOT/"src"))
from qqq_sim.config import configure_environment
from qqq_sim.serialization import save_json
from qqq_sim._build_info import BUILD_ID


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect():
    configure_environment()
    dependencies=[]
    notices=[]
    runtime_notices=[]
    python_license=Path(sys.base_prefix)/'LICENSE.txt'
    if not python_license.is_file():
        raise RuntimeError('The exact embedded Python runtime license is missing')
    license_paths=[python_license,*sorted((ROOT/'packaging/licenses').glob('*'))]
    for path in license_paths:
        if path.is_file():
            content=path.read_text(encoding='utf-8')
            runtime_notices.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
            notices.append(f'\n{"="*72}\nRuntime license: {path.name}\n{"="*72}\n{content}')
    for dist in sorted(md.distributions(),key=lambda d:d.metadata["Name"].lower()):
        if dist.metadata["Name"]=="qqq-simulator":
            continue
        name=dist.metadata["Name"]
        record={"name":name,"version":dist.version,"license_expression":dist.metadata.get("License-Expression"),
                "license_metadata":dist.metadata.get("License"),"homepage":dist.metadata.get("Home-page"),"license_files":[]}
        for item in dist.files or []:
            if re.search(r"(^|/)(licenses?/|copying|licen[cs]e|notice)",str(item),re.I) and not str(item).endswith((".py",".pyc",".pyd")):
                path=Path(dist.locate_file(item))
                if path.is_file() and path.stat().st_size<2_000_000:
                    try:
                        text=path.read_text(encoding="utf-8")
                    except (UnicodeError,OSError):
                        continue
                    record["license_files"].append({"path":str(item),"sha256":sha(path)})
                    notices.append(f"\n{'='*72}\n{name} {dist.version} — {item}\n{'='*72}\n{text}")
        dependencies.append(record)
    tests_path=ROOT/"runtime/results/tests.xml"
    tests={"status":"NOT_RUN"}
    if tests_path.exists():
        xml=ET.parse(tests_path)
        suites=xml.findall(".//testsuite")
        tests={"tests":sum(int(s.attrib.get("tests",0)) for s in suites),
               "failures":sum(int(s.attrib.get("failures",0)) for s in suites),
               "errors":sum(int(s.attrib.get("errors",0)) for s in suites),
               "skipped":sum(int(s.attrib.get("skipped",0)) for s in suites),
               "seconds":sum(float(s.attrib.get("time",0)) for s in suites),
               "artifact_sha256":sha(tests_path)}
        tests["status"]="PASS" if tests["failures"]+tests["errors"]==0 else "FAIL"
    source={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in sorted((ROOT/"src").rglob("*.py"))}
    account_output=subprocess.check_output([str(Path(os.environ['SystemRoot'])/'System32/whoami.exe'),'/user','/fo','csv','/nh'],text=True,encoding='utf-8',errors='replace')
    account_sid=list(csv.reader(io.StringIO(account_output)))[0][-1].strip()
    if not account_sid.startswith('S-1-'):raise RuntimeError('Windows account identity could not be recorded')
    account_hash=hashlib.sha256(account_sid.encode('ascii')).hexdigest()
    manifest={"schema_version":1,"recorded_at_utc":datetime.now(timezone.utc).isoformat(),
              "project_version":md.version("qqq-simulator"),"build_id":BUILD_ID,"python":sys.version,"executable":sys.executable,
              "virtual_environment":sys.prefix,"platform":platform.platform(),
              "dependencies":dependencies,"runtime_license_files":runtime_notices,"source_sha256":source,"source_count":len(source),
              "specification_sha256":sha(ROOT/"qqq_simulator_5_part_implementation_plan_complete.txt"),
              "dependency_lock_sha256":sha(ROOT/"requirements.lock"),"tests":tests,"real_orders_sent":False,
              "windows_clean_account":"NOT_RUN","build_account_identity_sha256":account_hash,"broker_parity":"NOT_CLAIMED"}
    save_json("runtime/results/release-manifest.json",manifest)
    (ROOT/"packaging/THIRD_PARTY_NOTICES.txt").write_text("Dependency license texts from the exact installed distributions.\n"+"\n".join(notices),encoding="utf-8")
    save_json("packaging/dependency-notices.json",dependencies)
    print(json.dumps({"tests":tests,"source_modules":len(source),"dependency_distributions":len(dependencies),"notices":len(notices)}))


if __name__=="__main__":
    collect()
