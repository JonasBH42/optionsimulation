"""Exercise a synthetic trade, pack selection, restart and export through Qt."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
import json
from PySide6.QtCore import QTimer, QRect, QPoint
from PySide6.QtWidgets import QApplication

from qqq_sim.config import AppConfig, writable_path
from qqq_sim.cli import demo_config
from qqq_sim.serialization import digest, save_json
from qqq_sim.replay.calendar import local_datetime
from .main_window import MainWindow


def smoke(output="runtime/results/desktop-smoke",pack_path="runtime/model_packs/validation_fixture"):
    output=writable_path(output)
    output.mkdir(parents=True,exist_ok=True)
    config=replace(demo_config(AppConfig()),runtime_root=str(output/"session_runtime"))
    app=QApplication.instance() or QApplication([])
    from qqq_sim.app import configure_fonts
    configure_fonts(app)
    window=MainWindow(config,local_datetime(datetime(2026,11,9,10,1)))
    window.show()
    started=perf_counter()
    result={"status":"FAILED","reason":"TIMEOUT","scale":app.primaryScreen().devicePixelRatio(),"synthetic":True}
    timer=QTimer()
    timer.setInterval(100)
    stage=0
    opened={}
    frozen={}
    errors=[]
    geometry_checks=[]
    window.worker.failed.connect(lambda kind,error:errors.append({"command":kind,"error":error}))

    def finish(record):
        nonlocal result
        result=record
        timer.stop()
        save_json(output/"result.json",result)
        window.close()

    def geometry_record():
        frame=window.frameGeometry()
        available=window.screen().availableGeometry()
        record={"frame":[frame.x(),frame.y(),frame.width(),frame.height()],
                "available":[available.x(),available.y(),available.width(),available.height()],
                "device_pixel_ratio":window.devicePixelRatioF(),"fits_available_screen":available.contains(frame)}
        if not record["fits_available_screen"]:
            raise AssertionError("Native window frame exceeds the available screen: "+str(record))
        return record

    def capture(name,panel="ticket",show_submit=False):
        window.chain_dock.raise_()
        dock={"ticket":window.ticket_dock,"account":window.account_dock,"risk":window.risk_dock}[panel]
        dock.raise_()
        app.processEvents()
        window.fit_to_screen()
        app.processEvents()
        if show_submit:
            window.ticket_scroll.ensureWidgetVisible(window.submit_button,8,8)
            app.processEvents()
            button=QRect(window.submit_button.mapTo(window.ticket_scroll.viewport(),QPoint()),window.submit_button.size())
            if not window.ticket_scroll.viewport().rect().contains(button):
                raise AssertionError("Submit control cannot be reached through the ticket scroll area")
        check=geometry_record()
        check.update(screenshot=name,panel=panel,submit_reachable=show_submit)
        geometry_checks.append(check)
        path=output/name
        if not window.grab().save(str(path)):
            raise RuntimeError("Screenshot save failed")
        return str(path)

    def check():
        nonlocal stage,opened,frozen
        try:
            if errors:
                raise RuntimeError(str(errors))
            if perf_counter()-started>60:
                raise TimeoutError(f"GUI smoke timed out at stage {stage}")
            if window.busy or window.bundle is None:
                return
            if stage==0:
                window.send("load_pack",path=str(Path(pack_path).resolve()))
                stage=1
            elif stage==1:
                rows=window.chain_model.rows
                for strike,side in ((600,-1),(595,1)):
                    row=next(i for i,e in enumerate(rows) if e.contract.right=="P" and e.contract.strike==strike)
                    window.chain.selectRow(window.chain_proxy.mapFromSource(window.chain_model.index(row,0)).row())
                    window.add_selected(side)
                window.debounce.stop()
                window.analyze()
                stage=10
            elif stage==10:
                if window._analysis_futures or window._analysis_ready:
                    return
                if window.risk.toPlainText().startswith("Calculating"):
                    return
                opened["analysis_display"]=window.risk.toPlainText()
                opened["risk_screenshot"]=capture("desktop-risk.png",panel="risk")
                opened["draft_screenshot"]=capture("desktop-ticket.png",show_submit=True)
                window.draft_action("submit")
                stage=2
            elif stage==2:
                window.step()
                stage=3
            elif stage==3:
                if len(window.bundle.positions)!=2:
                    raise AssertionError("Synthetic spread did not fill both legs")
                opened={**opened,"positions":window.bundle.positions,"account":window.bundle.account,
                        "screenshot":capture("desktop.png",show_submit=True),
                        "account_screenshot":capture("desktop-account.png",panel="account")}
                window.select_campaign("CLOSE")
                window.debounce.stop()
                window.draft_action("submit")
                stage=4
            elif stage==4:
                window.step()
                stage=5
            elif stage==5:
                if window.bundle.positions:
                    raise AssertionError("Closing order did not flatten the synthetic spread")
                if len(window.bundle.orders)!=2 or any(x["status"]!="FILLED" for x in window.bundle.orders):
                    raise AssertionError("Opening and closing orders must each fill exactly once")
                frozen={"account":window.bundle.account,"orders":window.bundle.orders,
                        "campaigns":window.bundle.campaigns,"version":window.bundle.version}
                window.send("export",path=str(output/"session-export.json"))
                stage=6
            elif stage==6:
                exported=json.loads((output/"session-export.json").read_text(encoding="utf-8"))
                if not exported.get("analyses"):
                    raise AssertionError("The frozen analysis request and result were not exported")
                session_path=Path(window.config.runtime_root)/"sessions"/window.bundle.session_id/"session.sqlite"
                window.send("restore",path=str(session_path))
                stage=7
            elif stage==7:
                restored={"account":window.bundle.account,"orders":window.bundle.orders,
                          "campaigns":window.bundle.campaigns,"version":window.bundle.version}
                if digest(restored)!=digest(frozen):
                    save_json(output/"restore-mismatch.json",{"before":frozen,"after":restored})
                    raise AssertionError("Restart changed the committed account or orders")
                closed_screenshot=capture("desktop-closed.png",show_submit=True)
                closed_account_screenshot=capture("desktop-closed-account.png",panel="account")
                finish({"status":"OK","scale":window.devicePixelRatioF(),"synthetic":True,
                        "state_version":window.bundle.version,"snapshot_id":window.bundle.snapshot.snapshot_id,
                        "opened":opened,"closed_account":window.bundle.account,"screenshot":closed_screenshot,
                        "closed_account_screenshot":closed_account_screenshot,"geometry_checks":geometry_checks,
                        "window_geometry":geometry_record(),
                        "pack_loaded":window.config.pack_path,"saved_and_restored":True,"export":str(output/"session-export.json"),
                        "elapsed_seconds":perf_counter()-started,"clean_windows_account":False})
        except Exception as exc:
            finish({"status":"FAILED","reason":type(exc).__name__,"detail":str(exc),"stage":stage,
                    "elapsed_seconds":perf_counter()-started,"synthetic":True})
    timer.timeout.connect(check)
    timer.start()
    app.exec()
    return result
