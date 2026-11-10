from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqq_sim.config import AppConfig
from qqq_sim.services import startup


def config_at(tmp_path):
    underlying,options=tmp_path/"underlying",tmp_path/"options"
    underlying.mkdir();options.mkdir()
    (underlying/"sample.txt").write_bytes(b"read-only source sample")
    (options/"sample.txt").write_bytes(b"read-only options sample")
    return replace(AppConfig(),runtime_root=str(tmp_path/"runtime"),underlying_root=str(underlying),options_root=str(options))


def test_real_import_timezone_runtime_probes_and_missing_pack_do_not_block_browsing(tmp_path):
    config=config_at(tmp_path)
    before=(Path(config.underlying_root)/"sample.txt").read_bytes()
    report=startup.startup_report(config)
    assert report["runtime_ready"] and report["can_open_session"]
    assert report["status"]=="LIMITED" and report["pack"]["status"]=="PACK_MISSING"
    assert not report["pack"]["analytics_verified"]
    assert set(report["dependencies"])==set(startup.DEPENDENCIES)
    assert all(check["status"]=="PASS" for check in report["checks"])
    assert {item["field"] for item in report["issues"]}=={"pack_path"}
    assert (Path(config.underlying_root)/"sample.txt").read_bytes()==before
    assert not list(Path(config.runtime_root).rglob("*.probe"))
    assert {"build_id","dependencies","python_executable","venv"}<=report.keys()


def test_missing_native_import_is_detected_even_with_installed_metadata(tmp_path,monkeypatch):
    config=config_at(tmp_path)
    original=startup.importlib.import_module
    def actual_import(name):
        if name=="scipy.special":raise ImportError("DLL load failed in test")
        return original(name)
    monkeypatch.setattr(startup.importlib,"import_module",actual_import)
    report=startup.startup_report(config)
    assert not report["runtime_ready"] and report["status"]=="SETUP_REQUIRED"
    assert any(item["name"]=="import:scipy" and "DLL load" in item["message"] for item in report["issues"])
    assert not report["network_access_attempted"]


def test_write_failure_low_disk_and_unreadable_source_have_specific_guidance(tmp_path,monkeypatch):
    config=replace(config_at(tmp_path),underlying_root=str(tmp_path/"absent"))
    original=startup._probe_write
    def write(path):
        if Path(path).name=="sessions":raise PermissionError("session directory denied")
        return original(path)
    monkeypatch.setattr(startup,"_probe_write",write)
    monkeypatch.setattr(startup.shutil,"disk_usage",lambda path:SimpleNamespace(free=1,total=100,used=99))
    report=startup.startup_report(config)
    assert not report["runtime_ready"] and not report["can_open_session"]
    guidance=startup.setup_guidance(report)
    assert "underlying_root" in guidance and "DISK_SPACE_LOW" in guidance and "runtime_root" in guidance
    assert "options_root:" not in guidance
    assert "or select Session > Open synthetic demonstration" in guidance


def test_source_only_failure_keeps_desktop_available_for_demo(tmp_path):
    config=replace(config_at(tmp_path),underlying_root=str(tmp_path/"missing"))
    report=startup.startup_report(config)
    assert report["runtime_ready"] and not report["can_open_session"]
    assert report["status"]=="LIMITED"


def test_pack_metadata_never_claims_full_array_qualification(tmp_path):
    from qqq_sim.risk.contracts import stable_hash
    config=config_at(tmp_path);folder=tmp_path/"pack";folder.mkdir()
    manifest={"schema_version":"QQQ_P50_PACK_V1","pack_id":"TEST_METADATA_ONLY","artifacts":{}}
    validation={"manifest_hash":stable_hash(manifest),"capabilities":{"hold":{"qualified":True}}}
    for name,value in (("manifest",manifest),("validation",validation)):
        raw=json.dumps(value).encode();(folder/(name+".json")).write_bytes(raw)
        (folder/(name+".sha256")).write_text(hashlib.sha256(raw).hexdigest(),encoding="ascii")
    report=startup.startup_report(replace(config,pack_path=str(folder)))
    assert report["pack"]["status"]=="PACK_AVAILABLE_PENDING_FULL_VERIFICATION"
    assert report["pack"]["available"] and not report["pack"]["analytics_verified"]
    assert report["status"]=="READY"
    (folder/"manifest.json").write_text("{}")
    changed=startup.startup_report(replace(config,pack_path=str(folder)))
    assert changed["runtime_ready"] and changed["pack"]["status"]=="PACK_INTEGRITY_FAILURE"


@pytest.mark.parametrize("can_open",[False,True])
def test_actual_app_launch_uses_nonmodal_preflight_and_preserves_synthetic_mode(qtbot,tmp_path,monkeypatch,can_open):
    from PySide6.QtWidgets import QApplication,QMainWindow,QDockWidget,QPlainTextEdit
    from qqq_sim import app
    from qqq_sim.ui import main_window
    config=config_at(tmp_path)
    report=startup.startup_report(config)
    report["can_open_session"]=can_open
    monkeypatch.setattr(startup,"startup_report",lambda supplied:report)
    monkeypatch.setattr(app,"configure_fonts",lambda application:None)
    monkeypatch.setattr(QApplication,"exec",lambda self:0)
    captured=[]
    class Window(QMainWindow):
        def __init__(self,supplied,at,synthetic=False,auto_open=True):
            super().__init__();captured.append((self,supplied,synthetic,auto_open))
    monkeypatch.setattr(main_window,"MainWindow",Window)
    assert app.launch(config,synthetic=True)==0
    window,supplied,synthetic,auto_open=captured[0]
    qtbot.addWidget(window)
    assert supplied is config and synthetic and auto_open==can_open
    assert window.startup_report is report
    dock=window.findChild(QDockWidget,"Startup checks")
    assert dock is not None and "PACK_MISSING" in dock.findChild(QPlainTextEdit).toPlainText()
    window.close()


def test_doctor_uses_shared_report_and_keeps_pack_optional(tmp_path,monkeypatch,capsys):
    from qqq_sim.cli import main
    config=config_at(tmp_path);path=tmp_path/"config.json";config.save(path)
    report=startup.startup_report(config)
    calls=[]
    def inspect(supplied):calls.append(supplied);return report
    monkeypatch.setattr(startup,"startup_report",inspect)
    assert main(["--config",str(path),"doctor"])==0
    printed=json.loads(capsys.readouterr().out)
    assert len(calls)==1 and printed["build_id"]==report["build_id"]
    assert printed["pack"]["status"]=="PACK_MISSING" and printed["runtime_ready"]


def test_blocking_setup_window_never_opens_a_session(qtbot,tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication,QPlainTextEdit
    from qqq_sim import app
    from qqq_sim.ui import main_window
    config=config_at(tmp_path);report=startup.startup_report(config)
    report.update(runtime_ready=False,status="SETUP_REQUIRED",can_open_session=False)
    report["issues"].append({"name":"disk_space","reason":"DISK_SPACE_LOW","required_for_runtime":True,"field":"runtime_root","remedy":"Free space."})
    monkeypatch.setattr(startup,"startup_report",lambda supplied:report)
    monkeypatch.setattr(app,"configure_fonts",lambda application:None)
    shown=[]
    def event_loop(self):
        shown.extend(widget for widget in self.topLevelWidgets() if widget.windowTitle()=="QQQ Simulator: setup required")
        return 0
    monkeypatch.setattr(QApplication,"exec",event_loop)
    def forbidden(*args,**kwargs):raise AssertionError("A session window must not open")
    monkeypatch.setattr(main_window,"MainWindow",forbidden)
    assert app.launch(config)==2 and len(shown)==1
    qtbot.addWidget(shown[0])
    assert "DISK_SPACE_LOW" in shown[0].findChild(QPlainTextEdit).toPlainText()
    shown[0].close()


def test_export_saved_cli_routes_readonly_api_without_replay_restore(tmp_path,monkeypatch,capsys):
    import sys
    from qqq_sim.cli import main
    config=config_at(tmp_path);path=tmp_path/"config.json";config.save(path)
    output=tmp_path/"saved.json";calls=[]
    def export_saved(source,destination,branch_id=None):
        calls.append((source,destination,branch_id));return output
    monkeypatch.setitem(sys.modules,"qqq_sim.services.saved_session_export",SimpleNamespace(export_saved_session=export_saved))
    assert main(["--config",str(path),"export-saved","--session","old-session","--branch","saved-branch","--output",str(output)])==0
    assert calls==[(Path(config.runtime_root)/"sessions/old-session/session.sqlite",str(output),"saved-branch")]
    assert json.loads(capsys.readouterr().out)["mode"]=="READ_ONLY_SAVED_STATE_EXPORT"
