"""Standalone runtime tests with development paths removed from child processes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import psutil

ROOT=Path(r"D:\optionsimulation")
sys.path.insert(0,str(ROOT/"src"))
from qqq_sim.serialization import save_json
from build_source_manifest import current


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--executable",default="dist/QQQSimulator.dist/QQQSimulator.exe")
    parser.add_argument("--platform",default="offscreen",choices=("offscreen","windows"))
    parser.add_argument("--pack",default="runtime/model_packs/validation_fixture_v2")
    args=parser.parse_args()
    executable=(ROOT/args.executable).resolve()
    executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest()
    env=dict(os.environ)
    for name in ("PYTHONPATH","PYTHONHOME","VIRTUAL_ENV","QT_PLUGIN_PATH",
                 "QT_SCALE_FACTOR","QT_SCREEN_SCALE_FACTORS","QT_ENABLE_HIGHDPI_SCALING",
                 "QT_FONT_DPI","QT_SCALE_FACTOR_ROUNDING_POLICY","QT_AUTO_SCREEN_SCALE_FACTOR"):
        env.pop(name,None)
    env.update(PATH=r"C:\Windows\System32;C:\Windows",TEMP=str(ROOT/".tmp"),TMP=str(ROOT/".tmp"),QT_QPA_PLATFORM=args.platform)
    records=[]
    artifact_root=f"runtime/results/standalone/{args.platform}"
    if args.platform == "offscreen":
        screen_config=f"{artifact_root}/screen.json"
        save_json(screen_config,{"windowFrameMargins":True,"screens":[{
            "name":"Smoke","x":0,"y":0,"width":2560,"height":1440,
            "availableGeometry":{"x":0,"y":0,"width":2560,"height":1390},
            "logicalDpi":96,"logicalBaseDpi":96,"dpr":1.0}]})
        env["QT_QPA_PLATFORM"]=f"offscreen:configfile={screen_config}"
    def run(command,scale=None,expected=0):
        child_env=dict(env)
        if scale: child_env["QT_SCALE_FACTOR"]=str(scale)
        started=time.perf_counter()
        p=subprocess.Popen([str(executable),*command],cwd=ROOT,env=child_env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="utf-8",errors="replace")
        samples=[]
        stop=threading.Event()
        def monitor():
            while not stop.is_set():
                try:
                    parent=psutil.Process(p.pid)
                    samples.append(sum(x.memory_info().rss for x in [parent,*parent.children(recursive=True)] if x.is_running()))
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    pass
                stop.wait(.05)
        observer=threading.Thread(target=monitor,daemon=True)
        observer.start()
        timed_out=False
        try:
            stdout,stderr=p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:
                for child in psutil.Process(p.pid).children(recursive=True):
                    child.kill()
            except psutil.NoSuchProcess:
                pass
            p.kill()
            stdout,stderr=p.communicate()
        finally:
            stop.set()
            observer.join(1)
        record={"command":command,"scale":scale,"returncode":p.returncode,"passed":not timed_out and p.returncode==expected,
                "stdout":stdout,"stderr":stderr,"seconds":time.perf_counter()-started,
                "sampled_aggregate_peak_rss_bytes":max(samples,default=0),"memory_sample_interval_seconds":.05,"timed_out":timed_out}
        records.append(record)
        return record
    doctor=run(["doctor"])
    try:
        version=json.loads(doctor["stdout"])
    except ValueError:
        version={}
    expected_build=current()[0]
    sources_match=version.get("build_id")==expected_build
    fast=run(["fast-query",str(Path(args.pack)/"example_reference_job.json"),
              "--pack",args.pack,"--output",f"{artifact_root}/fast-query.json"],expected=2)
    try:
        fast_result=json.loads(fast["stdout"])
        fast["passed"] = fast["passed"] and fast_result.get("status")=="UNSUPPORTED" and fast_result.get("reason")=="PACK_NOT_QUALIFIED"
    except ValueError:
        fast["passed"]=False
    validation=run(["validate-pack",args.pack,"--output",f"{artifact_root}/validation.json"],expected=2)
    try:
        validation_result=json.loads(validation["stdout"])
        validation["passed"]=validation["passed"] and validation_result.get("status")=="VALID_UNQUALIFIED_PACK" and validation_result.get("online_import_boundary",{}).get("passed") is True
    except ValueError:
        validation["passed"]=False
    reference_source=json.loads((ROOT/"runtime/risk_validation/qualified_far_otm_reference_parallel.json").read_text(encoding="utf-8"))
    reference_job=f"{artifact_root}/reference-job.json"
    save_json(reference_job,reference_source["job"])
    reference=run(["reference",reference_job,"--output",f"{artifact_root}/reference.json",
                   "--checkpoint",f"{artifact_root}/reference-{expected_build[:16]}.checkpoint.json"])
    try:
        reference_result=json.loads(reference["stdout"])
        reference["passed"]=reference["passed"] and reference_result.get("release_qualified") is True
    except ValueError:
        reference["passed"]=False
    run(["--demo","snapshot","--at","2026-11-09T10:01:00-05:00","--output",f"{artifact_root}/snapshot.json"])
    replay=run(["--demo","replay","--at","2026-11-09T10:01:00-05:00","--minutes","5","--output",f"{artifact_root}/replay.json"])
    try:
        session=json.loads(replay["stdout"])["session_id"]
        run(["export",session,"--output",f"{artifact_root}/export.json"])
    except (ValueError,KeyError):
        records.append({"command":["export"],"passed":False,"reason":"REPLAY_DID_NOT_RETURN_SESSION_ID"})
    def desktop(label,factor,target_dpr=None):
        record=run(["desktop-smoke","--pack",args.pack,"--output",f"{artifact_root}/desktop-{label}"],factor)
        try:
            result=json.loads(record["stdout"])
            actual_dpr=float(result["scale"])
            geometry=result["window_geometry"]
            checks=result["geometry_checks"]
            record.update(actual_device_pixel_ratio=actual_dpr,target_device_pixel_ratio=target_dpr,
                          window_geometry=geometry,geometry_checks=checks)
            record["passed"]=record["passed"] and geometry["fits_available_screen"] is True and bool(checks)
            record["passed"]=record["passed"] and all(check["fits_available_screen"] is True for check in checks)
            record["passed"]=record["passed"] and {"ticket","account","risk"}.issubset({check["panel"] for check in checks})
            if target_dpr is not None:
                record["passed"]=record["passed"] and abs(actual_dpr-target_dpr)<.01
            return actual_dpr
        except (ValueError,KeyError,TypeError):
            record["passed"]=False
            return None
    native_dpr=desktop("native",1)
    if native_dpr is not None and native_dpr>0:
        for target_dpr in (1,1.5,2):
            desktop(str(int(target_dpr*100)),target_dpr/native_dpr,target_dpr)
    else:
        records.append({"command":["desktop-smoke"],"passed":False,"reason":"NATIVE_DEVICE_PIXEL_RATIO_UNAVAILABLE"})
    executable_unchanged=hashlib.sha256(executable.read_bytes()).hexdigest()==executable_sha256
    report={"status":"PASS" if all(x["passed"] for x in records) and sources_match and executable_unchanged else "FAIL", "executable":str(executable),
            "executable_sha256":executable_sha256,"executable_unchanged_during_checks":executable_unchanged,
            "actual_build_id":version.get("build_id"),"expected_build_id":expected_build,"sources_match":sources_match,
            "development_paths_removed":True,"platform":args.platform,"native_device_pixel_ratio":native_dpr,"clean_windows_account":False,
            "clean_account_reason":"Tests ran on the build Windows account; this is not evidence of a separate clean account.","commands":records}
    save_json(f"runtime/results/windows-standalone-smoke-{args.platform}.json",report)
    save_json("runtime/results/windows-standalone-smoke.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="commands"},indent=2))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__": raise SystemExit(main())
