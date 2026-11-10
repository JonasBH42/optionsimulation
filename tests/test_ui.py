from datetime import datetime
from decimal import Decimal
import pytest
from PySide6.QtCore import Qt, QModelIndex
from qqq_sim.config import AppConfig
from qqq_sim.cli import demo_config, make_data
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.ui.chain import ChainTableModel, ChainProxy
from qqq_sim.ui.chart import CandlestickItem
from qqq_sim.ui.main_window import MainWindow


def test_chain_roles_sort_and_visible_chart(qtbot,tmp_path):
    config=demo_config(AppConfig(),tmp_path/"data")
    snapshot=make_data(config).snapshot(local_datetime(datetime(2026,11,9,10,1)))
    model=ChainTableModel()
    model.apply(snapshot.quotes,1,snapshot.snapshot_id)
    assert model.rowCount(model.index(0,0))==0
    assert model.columnCount(model.index(0,0))==0
    assert model.data(QModelIndex()) is None
    proxy=ChainProxy()
    proxy.setSourceModel(model)
    proxy.sort(2,Qt.SortOrder.AscendingOrder)
    strikes=[proxy.index(i,2).data(Qt.ItemDataRole.UserRole) for i in range(proxy.rowCount())]
    assert strikes==sorted(strikes)
    updates=[]
    model.dataChanged.connect(lambda left,right,roles:updates.append((model.state_version,model.snapshot_id,roles)))
    model.apply(snapshot.quotes,2,"changed-snapshot")
    assert updates[-1][:2]==(2,"changed-snapshot")
    assert Qt.ItemDataRole.ForegroundRole in updates[-1][2]
    item=CandlestickItem()
    item.set_candles(snapshot.candles)
    assert item.boundingRect().right()<snapshot.asof.timestamp()
    assert len(item.chunks)==1
    old=item.chunks[0]
    item.set_candles(snapshot.candles)
    assert item.chunks[0] is old


def test_desktop_synchronized_step_and_clean_shutdown(qtbot,tmp_path,monkeypatch):
    from dataclasses import replace
    config=replace(demo_config(AppConfig(),tmp_path/"data"),runtime_root=str(tmp_path/"runtime"))
    window=MainWindow(config,local_datetime(datetime(2026,11,9,10,1)))
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(lambda:window.bundle is not None and not window.busy,timeout=30000)
    first=window.bundle.version
    window.step()
    qtbot.waitUntil(lambda:window.bundle.version>first and not window.busy,timeout=30000)
    assert window.chain_model.state_version==window.bundle.version
    assert str(window.bundle.version) in window.clock_label.text()
    assert window.chart.visible[-1].available_at<=window.bundle.decision_at
    window.expiry.setCurrentText("2026-11-13")
    assert "4 calendar DTE" in window.expiry_info.text()
    original=window.bundle
    empty=replace(original.snapshot,candles=())
    zero_mark={"campaign_id":"zero-price","quantity":1,"mark":Decimal("0.00"),
               "contract":{"expiry":"2026-11-13","right":"P","strike":"595"}}
    window.apply_bundle(replace(original,snapshot=empty,positions=(zero_mark,)))
    assert "none visible" in window.clock_label.text()
    assert window.portfolio.item(0,3).text()=="0.00"
    window.apply_bundle(original)
    from concurrent.futures import Future
    from qqq_sim.services.analysis import AnalysisIdentity
    identity=AnalysisIdentity("s","b",1,"snap","selection","campaign","objective","pack","rule")
    window.analysis.gate.request(identity)
    previous=window.selection_version
    window.order_type.setCurrentText("LIMIT")
    assert window.selection_version>previous and not window.analysis.gate.accepts(identity)
    stale=Future()
    stale.set_exception(RuntimeError("stale numerical worker failure"))
    window.analysis.pending=stale
    window.analysis.pending_identity=identity
    window._analysis_futures[stale]=(identity,None,None)
    window.risk.setPlainText("Current state")
    window.poll_analysis()
    assert window.risk.toPlainText()=="Current state"
    assert not window._analysis_futures and window.analysis.pending is stale
    historical=Future()
    historical.set_result((identity,{"status":"UNSUPPORTED","reason":"PACK_NOT_QUALIFIED"}))
    window._analysis_futures[historical]=(identity,"frozen-request","original.sqlite")
    window.busy=True
    window.poll_analysis()
    assert window._analysis_ready==[{"identity":identity,"request_hash":"frozen-request",
                                   "archive_path":"original.sqlite",
                                   "result":{"status":"UNSUPPORTED","reason":"PACK_NOT_QUALIFIED"}}]
    assert window.risk.toPlainText()=="Current state"
    window._analysis_record_inflight=window._analysis_ready[0]
    window.failed('record_analysis',{'status':'FAILED','reason':'INJECTED_ARCHIVE_WRITE_ERROR'})
    assert len(window._analysis_ready)==1 and window._analysis_archive_blocked
    assert not window._flush_analysis_record()
    window._analysis_ready.clear()
    window.busy=False
    window.chain.selectRow(0)
    window.add_selected(1)
    window.debounce.stop()
    crashed=Future()
    crashed.set_exception(RuntimeError("worker crashed"))
    window._analysis_futures[crashed]=(identity,None,None)
    window.analysis.gate.request(identity)
    window.poll_analysis()
    assert window._analysis_retry_required
    previous=window.selection_version
    sent=[]
    with monkeypatch.context() as patch:
        patch.setattr(window,'send',lambda kind,**values:sent.append((kind,values)))
        window.analyze()
    assert window.selection_version==previous+1 and sent[0][0]=='analyze'
    assert sent[0][1]['selection_hash']==window.analysis_selection_hash()
    assert not window._analysis_retry_required
    window.risk.setPlainText('Current result')
    window.failed('analyze',{'selection_hash':window.analysis_selection_hash(),'state_version':window.bundle.version-1,
                             'status':'FAILED','reason':'STALE_ERROR'})
    assert window.risk.toPlainText()=='Current result'
    window.close()
    qtbot.waitUntil(lambda:not window.thread.isRunning(),timeout=10000)


def test_step_progress_keeps_diagnostics_at_displayed_state(qtbot,tmp_path,monkeypatch):
    from dataclasses import replace
    from datetime import timedelta
    from threading import Event
    import json
    config=replace(demo_config(AppConfig(),tmp_path/"data"),runtime_root=str(tmp_path/"runtime"))
    window=MainWindow(config,local_datetime(datetime(2026,11,9,10,1)))
    qtbot.addWidget(window)
    window.show()
    entered,release=Event(),Event()
    try:
        qtbot.waitUntil(lambda:window.bundle is not None and not window.busy,timeout=10000)
        displayed=window.bundle
        data=window.worker.controller.data
        original=data.snapshot
        def slow_snapshot(*args,**kwargs):
            entered.set()
            assert release.wait(10)
            return original(*args,**kwargs)
        monkeypatch.setattr(data,"snapshot",slow_snapshot)
        window.step_count.setValue(2)
        window.step()
        qtbot.waitUntil(lambda:entered.is_set() and "saved through" in window.statusBar().currentMessage(),timeout=5000)
        assert window.bundle is displayed and window.busy
        assert "Display and diagnostics:" in window.statusBar().currentMessage()
        assert json.loads(window.diagnostics.toPlainText())["decision_at"]==displayed.decision_at.isoformat()
        window.pause()
        assert "Stopping after the current minute" in window.statusBar().currentMessage()
        release.set()
        qtbot.waitUntil(lambda:not window.busy,timeout=10000)
        assert window.bundle.status=="CANCELED"
        assert window.bundle.decision_at==displayed.decision_at+timedelta(minutes=1)
        assert json.loads(window.diagnostics.toPlainText())["decision_at"]==window.bundle.decision_at.isoformat()
    finally:
        release.set()
        window.close()
        qtbot.waitUntil(lambda:not window.thread.isRunning(),timeout=10000)


@pytest.mark.parametrize('close_during_prepare',[False,True])
def test_shutdown_archives_prepared_and_running_requests(qtbot,tmp_path,monkeypatch,close_during_prepare):
    from concurrent.futures import Future
    from dataclasses import replace
    from threading import Event
    from qqq_sim.services.analysis import AnalysisIdentity
    from qqq_sim.services.analytic_records import register_analysis_request
    from qqq_sim.storage.events import EventStore
    config=demo_config(AppConfig(),tmp_path/'data')
    pack=tmp_path/'pack'; pack.mkdir(); (pack/'manifest.json').write_text('{}',encoding='utf-8')
    config=replace(config,runtime_root=str(tmp_path/'runtime'),pack_path=str(pack))
    entered,release=Event(),Event()
    if not close_during_prepare:
        release.set()
    def prepare(controller,draft,selection_hash,**kwargs):
        entered.set()
        assert release.wait(5)
        identity=AnalysisIdentity(controller.session_id,controller.branch_id,controller.version,
            controller.snapshot.snapshot_id,selection_hash,'campaign','objective',controller._pack_hash,'rule')
        return register_analysis_request(controller,{'identity':identity,'job':{},'pack_path':str(pack)},draft=draft)
    monkeypatch.setattr('qqq_sim.services.analysis_inputs.prepare_analysis',prepare)
    class Pool:
        submitted=0
        def submit(self,*args):
            self.submitted+=1
            future=Future(); future.set_running_or_notify_cancel()
            return future
        def shutdown(self,**kwargs): pass
    window=MainWindow(config,local_datetime(datetime(2026,11,9,10,1)))
    qtbot.addWidget(window); window.show()
    qtbot.waitUntil(lambda:window.bundle is not None and not window.busy,timeout=10000)
    version=window.bundle.version
    archive=window.worker.controller.store.path
    pool=Pool(); window.analysis.pool=pool
    window.chain.selectRow(0); window.add_selected(1); window.debounce.stop(); window.analyze()
    qtbot.waitUntil(entered.is_set,timeout=10000)
    if close_during_prepare:
        window.close()
        assert window.thread.isRunning()
        release.set()
    else:
        qtbot.waitUntil(lambda:bool(window._analysis_futures),timeout=10000)
        window.close()
    qtbot.waitUntil(lambda:not window.thread.isRunning(),timeout=10000)
    with EventStore(archive) as store:
        records=store.analysis_records()
        assert len(records)==1 and records[0]['status']=='CANCELED'
        assert records[0]['result_record']['payload']['result']['reason']=='APPLICATION_SHUTDOWN'
        assert store.restore(records[0]['identity']['branch_id'])['version']==version
    assert pool.submitted==(0 if close_during_prepare else 1)
