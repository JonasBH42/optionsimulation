"""Synchronized dockable desktop workspace for historical simulation."""
from __future__ import annotations
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
import json

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QDateTime, QByteArray, QPoint
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QMainWindow, QWidget, QDockWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QPushButton, QTableView, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFormLayout, QDoubleSpinBox, QDateTimeEdit, QLineEdit,
    QPlainTextEdit, QFileDialog, QToolBar, QSplitter, QInputDialog, QProgressBar,
    QScrollArea, QFrame, QSizePolicy)

from qqq_sim.config import writable_path
from qqq_sim.serialization import dumps, digest, save_json
from qqq_sim.broker.models import OrderLeg, OrderDraft, Contract
from qqq_sim.services.adapters import broker_contract
from qqq_sim.services.analysis import AnalysisCoordinator
from qqq_sim.replay.calendar import CalendarService, local_datetime, NY
from qqq_sim.data.underlying import aggregate_candles
from .chain import ChainTableModel, ChainProxy
from .chart import ChartWidget
from .worker import SimulationWorker


def ny_editor(value=None):
    editor=QDateTimeEdit()
    editor.setDisplayFormat("yyyy-MM-dd HH:mm")
    editor.setCalendarPopup(True)
    editor.setDateTime(QDateTime.fromString((value or datetime(2026,11,9,10,1)).strftime("%Y-%m-%d %H:%M"),"yyyy-MM-dd HH:mm"))
    return editor


def editor_time(editor):
    return local_datetime(datetime.strptime(editor.dateTime().toString("yyyy-MM-dd HH:mm"),"%Y-%m-%d %H:%M"))


class MainWindow(QMainWindow):
    command = Signal(object)

    def __init__(self, config, initial_at=None, synthetic=False, auto_open=True):
        super().__init__()
        self.config=config
        self.bundle=None
        self.busy=False
        self._advance_command_id=None
        self._closing=False
        self._fit_screen=None
        self._screen_fit_pending=False
        self._screen_signal_bound=False
        self.draft_legs=[]
        self.analysis_current_position=False
        self._benchmark_requested=set()
        self._benchmark_futures={}
        self._benchmark_ready=[]
        self._benchmark_retries={}
        self._analysis_futures={}
        self._analysis_ready=[]
        self._analysis_retry_required=False
        self._analysis_record_inflight=None
        self._analysis_archive_blocked=False
        self._shutdown_requested=False
        self.selection_version=0
        self.selected_campaign=None
        from qqq_sim.services.calendars import load_calendar
        self.calendar=load_calendar(config.calendar_path)
        self.analysis=AnalysisCoordinator()
        self.setWindowTitle("QQQ Replay · Historical options simulator")
        self.resize(1560,980)
        self.setDockNestingEnabled(True)
        self.setStyleSheet("QMainWindow{background:#eef2f7} QDockWidget::title{background:#dce5ef;padding:7px;font-weight:600;} QTableView{gridline-color:#e7ebf2;alternate-background-color:#f6f8fb;} QPushButton{padding:5px 10px;} QToolBar{spacing:6px;padding:5px;} QPlainTextEdit{background:#fbfcfe;} ")
        self._toolbar(initial_at)
        self.chart=ChartWidget()
        self.chart.caption.setWordWrap(True)
        self.setCentralWidget(self.chart)
        self._chain()
        self._ticket()
        self._portfolio()
        self._records()
        self._menus()
        self.statusBar().showMessage("Opening read-only historical sources…")
        self.progress=QProgressBar()
        self.progress.setRange(0,0)
        self.progress.setMaximumWidth(140)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.play_timer=QTimer(self)
        self.play_timer.setInterval(600)
        self.play_timer.timeout.connect(lambda:self.step() if not self.busy else None)
        self.debounce=QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(200)
        self.debounce.timeout.connect(self.analyze)
        self.analysis_timer=QTimer(self)
        self.analysis_timer.setInterval(50)
        self.analysis_timer.timeout.connect(self.poll_analysis)
        self.analysis_timer.start()
        self.thread=QThread(self)
        self.worker=SimulationWorker(config)
        self.worker.moveToThread(self.thread)
        self.command.connect(self.worker.execute,Qt.ConnectionType.QueuedConnection)
        self.worker.completed.connect(self.completed,Qt.ConnectionType.QueuedConnection)
        self.worker.failed.connect(self.failed,Qt.ConnectionType.QueuedConnection)
        self.worker.advance_progress.connect(self.advance_progress,Qt.ConnectionType.QueuedConnection)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()
        self._restore_layout()
        if auto_open:
            QTimer.singleShot(0,lambda:self.send("demo" if synthetic else "open",at=initial_at))

    def dock(self,title,widget,area,scrollable=False):
        dock=QDockWidget(title,self)
        dock.setObjectName(title)
        if scrollable:
            scroll=QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setMinimumSize(100,80)
            scroll.setWidget(widget)
            widget=scroll
        dock.setWidget(widget)
        self.addDockWidget(area,dock)
        dock.topLevelChanged.connect(self._schedule_screen_fit)
        return dock

    def showEvent(self,event):
        super().showEvent(event)
        handle=self.windowHandle()
        if handle is not None and not self._screen_signal_bound:
            handle.screenChanged.connect(self._watch_screen)
            self._screen_signal_bound=True
        self._watch_screen(self.screen())

    def _watch_screen(self,screen):
        if self._fit_screen is not screen:
            if self._fit_screen is not None:
                try:
                    self._fit_screen.availableGeometryChanged.disconnect(self._schedule_screen_fit)
                except (RuntimeError,TypeError):
                    pass
            self._fit_screen=screen
            if screen is not None:
                screen.availableGeometryChanged.connect(self._schedule_screen_fit)
        self._schedule_screen_fit()

    def _schedule_screen_fit(self,*args):
        if self._closing or self._screen_fit_pending:
            return
        self._screen_fit_pending=True
        QTimer.singleShot(0,self.fit_to_screen)

    def fit_to_screen(self,available=None):
        """Keep native frames inside the current work area, including restored layouts."""
        self._screen_fit_pending=False
        if self._closing:
            return
        widgets=[self,*[dock for dock in self.findChildren(QDockWidget) if dock.isFloating()]]
        for widget in widgets:
            screen=widget.screen()
            if available is None and screen is None:
                continue
            bounds=(available if available is not None else screen.availableGeometry()).adjusted(4,4,-4,-4)
            if bounds.width()<=0 or bounds.height()<=0:
                continue
            frame=widget.frameGeometry()
            frame_width=max(0,frame.width()-widget.width())
            frame_height=max(0,frame.height()-widget.height())
            width=max(1,bounds.width()-frame_width)
            height=max(1,bounds.height()-frame_height)
            widget.setMaximumSize(width,height)
            widget.resize(min(widget.width(),width),min(widget.height(),height))
            if widget is self and height<1000 and not self.chain_dock.isFloating():
                self.resizeDocks([self.chain_dock],[max(190,int(height*.36))],Qt.Orientation.Vertical)
            if widget.layout() is not None:
                widget.layout().activate()
            frame=widget.frameGeometry()
            left=max(bounds.left(),min(frame.left(),bounds.right()-frame.width()+1))
            top=max(bounds.top(),min(frame.top(),bounds.bottom()-frame.height()+1))
            widget.move(widget.pos()+QPoint(left-frame.left(),top-frame.top()))

    def _toolbar(self, initial_at):
        bar=QToolBar("Replay",self)
        bar.setObjectName("Replay toolbar")
        self.addToolBar(bar)
        title=QLabel("  QQQ  ·  HISTORICAL REPLAY  ")
        title.setStyleSheet("font-weight:700;color:#274c75")
        bar.addWidget(title)
        self.play_button=QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        bar.addWidget(self.play_button)
        self.step_button=QPushButton("Step →")
        self.step_button.clicked.connect(self.step)
        bar.addWidget(self.step_button)
        self.step_count=QSpinBox()
        self.step_count.setRange(1,10000)
        self.step_count.setValue(1)
        bar.addWidget(self.step_count)
        self.step_unit=QComboBox()
        self.step_unit.addItems(["trading minutes","elapsed minutes"])
        bar.addWidget(self.step_unit)
        bar.addSeparator()
        self.jump_time=ny_editor(initial_at)
        bar.addWidget(self.jump_time)
        jump=QPushButton("Jump ET")
        jump.clicked.connect(lambda:self.send("advance",target=editor_time(self.jump_time)))
        bar.addWidget(jump)
        next_session=QPushButton("Next session")
        next_session.clicked.connect(self.next_session)
        bar.addWidget(next_session)
        bar.addSeparator()
        bar.addWidget(QLabel("Chart"))
        self.aggregation=QComboBox()
        self.aggregation.addItems(["1 min","5 min","15 min","30 min","60 min","Daily"])
        self.aggregation.currentIndexChanged.connect(self.redraw_chart)
        bar.addWidget(self.aggregation)
        self.clock_label=QLabel("No committed state")
        self.clock_label.setWordWrap(True)
        self.clock_label.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Preferred)
        self.clock_label.setStyleSheet("padding:5px;font-weight:600")
        secondary=QToolBar("State",self)
        secondary.setObjectName("State toolbar")
        secondary.addWidget(self.clock_label)
        self.addToolBarBreak()
        self.addToolBar(secondary)

    def _chain(self):
        widget=QWidget()
        layout=QVBoxLayout(widget)
        row=QHBoxLayout()
        row.addWidget(QLabel("Expiration"))
        self.expiry=QComboBox()
        self.expiry.setMinimumWidth(140)
        self.expiry.currentTextChanged.connect(lambda text:self.chain_proxy.set_expiry(None if text=="All observed" else text))
        self.expiry.currentTextChanged.connect(self.update_expiry_info)
        row.addWidget(self.expiry)
        self.expiry_info=QLabel()
        self.expiry_info.setWordWrap(True)
        row.addWidget(self.expiry_info,1)
        actions=QHBoxLayout()
        actions.addStretch()
        for label,side in (("Sell selected (bid)",-1),("Buy selected (ask)",1)):
            button=QPushButton(label)
            button.clicked.connect(lambda checked=False,side=side:self.add_selected(side))
            actions.addWidget(button)
        layout.addLayout(row)
        layout.addLayout(actions)
        self.chain_model=ChainTableModel(self)
        self.chain_proxy=ChainProxy(self)
        self.chain_proxy.setSourceModel(self.chain_model)
        self.chain=QTableView()
        self.chain.setModel(self.chain_proxy)
        self.chain.setSortingEnabled(True)
        self.chain.setAlternatingRowColors(True)
        self.chain.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.chain.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.chain.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.chain.verticalHeader().hide()
        self.chain.doubleClicked.connect(lambda i:self.add_selected(-1 if i.column()==3 else 1))
        layout.addWidget(self.chain)
        self.chain_dock=self.dock("Observed options chain",widget,Qt.DockWidgetArea.BottomDockWidgetArea)

    def _ticket(self):
        widget=QWidget()
        layout=QVBoxLayout(widget)
        label=QLabel("Simulated orders only · protection model: simultaneous delivery hedge\nFixed rule counterfactual · no account-specific broker parity")
        label.setWordWrap(True)
        layout.addWidget(label)
        self.ticket=QTableWidget(0,4)
        self.ticket.setHorizontalHeaderLabels(["Contract","Signed qty","Side","Quote age"])
        self.ticket.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        self.ticket.setColumnWidth(1,70)
        self.ticket.setColumnWidth(2,55)
        self.ticket.setColumnWidth(3,75)
        self.ticket.setMaximumHeight(170)
        layout.addWidget(self.ticket)
        edit_row=QHBoxLayout()
        undo=QPushButton("Undo leg")
        undo.clicked.connect(self.undo_leg)
        clear=QPushButton("Clear draft")
        clear.clicked.connect(self.clear_draft)
        edit_row.addWidget(undo)
        edit_row.addWidget(clear)
        layout.addLayout(edit_row)
        form=QFormLayout()
        self.campaign_edit=QLineEdit("campaign-"+uuid4().hex[:8])
        self.campaign_edit.setToolTip("Keep the same campaign ID through closes and rolls. A new ID does not reset the account loss cap.")
        form.addRow("Campaign",self.campaign_edit)
        self.intent=QComboBox()
        self.intent.addItems(["OPEN","CLOSE","ADJUST","ROLL"])
        form.addRow("Action",self.intent)
        self.order_type=QComboBox()
        self.order_type.addItems(["PROTECTED_LIMIT","LIMIT","MARKET"])
        form.addRow("Order type",self.order_type)
        self.limit=QDoubleSpinBox()
        self.limit.setRange(-1000000,1000000)
        self.limit.setDecimals(0)
        self.limit.setSingleStep(1)
        self.limit.setToolTip("Minimum signed net premium cash per ratio unit in USD. +100 requires at least $100 credit; -200 permits at most $200 debit. Fees are separate.")
        form.addRow("Min cash / ratio ($)",self.limit)
        self.fee_cap=QDoubleSpinBox()
        self.fee_cap.setRange(0,1000)
        self.fee_cap.setValue(50)
        form.addRow("Maximum fees ($)",self.fee_cap)
        self.deadline=ny_editor(datetime(2026,11,13,15,30))
        self.cutoff=ny_editor(datetime(2026,11,13,16,0))
        form.addRow("Target deadline · NY",self.deadline)
        form.addRow("Hard cutoff · NY",self.cutoff)
        self.tif=QComboBox()
        self.tif.addItems(["DAY","GTC"])
        form.addRow("Time in force",self.tif)
        layout.addLayout(form)
        self.ticket_summary=QLabel("Select exact contracts from the observed chain.")
        self.ticket_summary.setWordWrap(True)
        layout.addWidget(self.ticket_summary)
        row=QHBoxLayout()
        preview=QPushButton("Check costs / risk")
        preview.clicked.connect(lambda:self.draft_action("preview"))
        analyze=QPushButton("Analyze P50")
        analyze.clicked.connect(self.analyze)
        row.addWidget(preview)
        row.addWidget(analyze)
        layout.addLayout(row)
        self.submit_button=QPushButton("Submit simulated order")
        self.submit_button.setStyleSheet("background:#255a86;color:white;font-weight:600;padding:9px")
        self.submit_button.clicked.connect(lambda:self.draft_action("submit"))
        layout.addWidget(self.submit_button)
        layout.addStretch()
        self.ticket_dock=self.dock("Trade ticket",widget,Qt.DockWidgetArea.RightDockWidgetArea,scrollable=True)
        self.ticket_scroll=self.ticket_dock.widget()
        self.ticket_dock.setMinimumWidth(320)
        for editor in (self.limit,self.fee_cap):
            editor.valueChanged.connect(self.draft_changed)
        for editor in (self.deadline,self.cutoff):
            editor.dateTimeChanged.connect(self.draft_changed)
        self.campaign_edit.textChanged.connect(self.draft_changed)
        for editor in (self.intent,self.order_type,self.tif):
            editor.currentTextChanged.connect(self.draft_changed)

    def _portfolio(self):
        widget=QWidget()
        layout=QVBoxLayout(widget)
        self.portfolio=QTableWidget(0,4)
        self.portfolio.setHorizontalHeaderLabels(["Campaign","Contract","Quantity","Mark ($)"])
        self.portfolio.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.portfolio.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.portfolio.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.portfolio)
        row=QHBoxLayout()
        for label,action in (("Close campaign","CLOSE"),("Adjust / roll campaign","ROLL"),("Analyze campaign","ANALYZE")):
            b=QPushButton(label)
            b.clicked.connect(lambda checked=False,action=action:self.select_campaign(action))
            row.addWidget(b)
        layout.addLayout(row)
        stock_reduce=QPushButton("Reduce assigned stock with a limit order")
        stock_reduce.clicked.connect(self.reduce_stock)
        layout.addWidget(stock_reduce)
        self.orders=QTableWidget(0,5)
        self.orders.setHorizontalHeaderLabels(["Order","Campaign","Status","Filled","Reserved ($)"])
        self.orders.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.orders.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.orders)
        cancel=QPushButton("Cancel selected working order")
        cancel.clicked.connect(self.cancel_order)
        layout.addWidget(cancel)
        self.portfolio_dock=self.dock("Positions and orders",widget,Qt.DockWidgetArea.BottomDockWidgetArea,scrollable=True)
        self.tabifyDockWidget(self.chain_dock,self.portfolio_dock)
        self.chain_dock.raise_()

    def _records(self):
        self.account=QPlainTextEdit()
        self.account.setReadOnly(True)
        self.risk=QPlainTextEdit()
        self.risk.setReadOnly(True)
        self.risk.setPlainText("HOLD P50: unavailable · PACK_MISSING\nManaged executed-target probability: unavailable · PACK_MISSING\n\nUse Model → Load pack to select an offline-built, validated pack.")
        self.history=QPlainTextEdit()
        self.history.setReadOnly(True)
        self.diagnostics=QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        self.account_dock=self.dock("Account and capital",self.account,Qt.DockWidgetArea.RightDockWidgetArea)
        self.risk_dock=self.dock("P50 and risk envelope",self.risk,Qt.DockWidgetArea.RightDockWidgetArea)
        self.tabifyDockWidget(self.ticket_dock,self.account_dock)
        self.tabifyDockWidget(self.ticket_dock,self.risk_dock)
        self.ticket_dock.raise_()
        self.history_dock=self.dock("History",self.history,Qt.DockWidgetArea.BottomDockWidgetArea)
        diagnostic_dock=self.dock("Diagnostics",self.diagnostics,Qt.DockWidgetArea.BottomDockWidgetArea)
        self.tabifyDockWidget(self.portfolio_dock,self.history_dock)
        self.tabifyDockWidget(self.portfolio_dock,diagnostic_dock)
        self.chain_dock.raise_()

    def _menus(self):
        session=self.menuBar().addMenu("Session")
        actions=[("New session at selected time",lambda:self.send("open",at=editor_time(self.jump_time))),
                 ("Open saved session…",self.restore_session),("Open synthetic demonstration",lambda:self.send("demo",at=local_datetime(datetime(2026,11,9,10,1)))),
                 ("Branch from committed event…",self.branch),("Export frozen session…",self.export_session)]
        for label,callback in actions:
            session.addAction(label,callback)
        model=self.menuBar().addMenu("Model")
        model.addAction("Load pack…",self.load_pack)
        model.addAction("Load dated scenario inputs…",self.load_parameters)
        model.addAction("Offline build instructions",lambda:self.show_info("Model building is explicit and runs outside the GUI.\n\n.\\launch.ps1 build-model --build-config path\\build.json\n.\\launch.ps1 validate-pack runtime\\model_packs\\pack\n\nOnly a pack that passes its numerical/domain gates may report a supported probability."))
        view=self.menuBar().addMenu("View")
        for dock in self.findChildren(QDockWidget):
            view.addAction(dock.toggleViewAction())
        for key,callback in (("Right",self.step),("Space",self.toggle_play),("Ctrl+Z",self.undo_leg),("Escape",self.pause),("Ctrl+L",lambda:self.chain.setFocus())):
            action=QAction(self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            self.addAction(action)

    def send(self,kind,**kwargs):
        if self.busy and kind!="close":
            return
        if kind not in ("open","restore","demo","close","load_pack","load_parameters") and self.bundle is None:
            return
        self.busy=True
        self.progress.setRange(0,0)
        self.progress.show()
        self.submit_button.setEnabled(False)
        self.step_button.setEnabled(False)
        self.statusBar().showMessage(f"{kind.replace('_',' ').capitalize()}…")
        payload={"kind":kind,"id":uuid4().hex,**kwargs}
        self._advance_command_id=payload["id"] if kind=="advance" else None
        if self.bundle:
            payload["version"]=self.bundle.version
        self.command.emit(payload)

    @Slot(object)
    def advance_progress(self,record):
        if not self.busy or record["command_id"]!=self._advance_command_id:
            return
        target,committed=record["target"],record["decision_at"]
        self.statusBar().showMessage(
            f"Advancing to {target:%Y-%m-%d %H:%M %Z} | saved through {committed:%Y-%m-%d %H:%M}"
            f" | {record['elapsed_seconds']:.1f}s | Esc to stop after this minute."
            f" Display and diagnostics: {self.bundle.decision_at:%Y-%m-%d %H:%M}, state {self.bundle.version}.")

    @Slot(str,object)
    def completed(self,kind,result):
        self.busy=False
        self._advance_command_id=None
        self.progress.hide()
        self.submit_button.setEnabled(self.bundle is not None and not self.analysis_current_position)
        self.step_button.setEnabled(self.bundle is not None)
        if kind=="close":
            self.thread.quit()
            self.thread.wait(5000)
            self._closing=True
            self.close()
            return
        if kind=="record_analysis":
            if self._analysis_record_inflight is not None:
                request_hash=self._analysis_record_inflight["request_hash"]
                self._analysis_ready=[item for item in self._analysis_ready if item["request_hash"]!=request_hash]
                self._analysis_record_inflight=None
        if self._shutdown_requested:
            if kind=="analyze" and "identity" in result and result.get("cached_result") is None:
                self._queue_shutdown_analysis(result["identity"],result.get("request_hash"),result.get("archive_path"))
            self._continue_shutdown()
            return
        if hasattr(result,"snapshot"):
            self.apply_bundle(result)
            if kind=="submit":
                self.clear_draft(reset_campaign=False)
            self.statusBar().showMessage(f"{result.status} · decision {result.decision_at:%Y-%m-%d %H:%M:%S %Z} · state {result.version}")
            if result.status!="OK":
                self.pause()
        elif kind=="prepare_benchmark":
            key=(result["identity"].session_id,result["identity"].branch_id,result["benchmark_id"])
            self._benchmark_futures[key]=self.analysis.submit_benchmark(result)
            self.statusBar().showMessage("Calculating frozen initial-fill benchmark")
        elif kind=="analyze" and "identity" in result:
            identity=result["identity"]
            try:
                current_selection=self.analysis_selection_hash()
            except ValueError:
                current_selection=None
            if (self.bundle is None or identity.session_id!=self.bundle.session_id or identity.branch_id!=self.bundle.branch_id
                or identity.state_version!=self.bundle.version or identity.snapshot_hash!=self.bundle.snapshot.snapshot_id
                or identity.selection_hash!=current_selection):
                if (result.get("request_hash") and result.get("cached_result") is None
                    and not self._has_analysis_request(result["request_hash"])):
                    self._analysis_ready.append({"identity":identity,"request_hash":result["request_hash"],
                        "archive_path":result.get("archive_path"),
                        "result":{"status":"CANCELED","reason":"SELECTION_CHANGED_BEFORE_ANALYSIS",
                                  "hold_probability":None,"managed_probability":None}})
                return
            if self._has_analysis_request(result.get("request_hash")):
                self.analysis.gate.request(identity)
                for record in self._analysis_ready:
                    if record["request_hash"]==result.get("request_hash"):
                        self.risk.setPlainText(dumps(record["result"],pretty=True))
                        self.risk_dock.raise_()
                return
            if result.get("cached_result") is not None:
                self.analysis.gate.request(identity)
                self._analysis_retry_required=result["cached_result"].get("status")=="FAILED"
                self.risk.setPlainText(dumps(result["cached_result"],pretty=True))
                self.risk_dock.raise_()
                return
            future=self.analysis.submit(identity,result["job"],result["pack_path"],result.get("campaign"),result.get("universe",()),result.get("limits"),result.get("proposal"))
            self._analysis_futures[future]=(identity,result.get("request_hash"),result.get("archive_path"))
            self.risk.setPlainText(f"Calculating for state {result['identity'].state_version}…")
        else:
            if kind=="analyze" and result.get("selection_hash"):
                try:
                    if (result["selection_hash"]!=self.analysis_selection_hash() or self.bundle is None
                        or result.get("state_version")!=self.bundle.version):
                        return
                except ValueError:
                    return
            self.statusBar().showMessage(f"{kind}: {result.get('status','saved')}")
            if kind in ("preview","analyze","load_pack","load_parameters"):
                self.risk.setPlainText(dumps(result,pretty=True))
                self.risk_dock.raise_()
            elif kind=="export":
                self.statusBar().showMessage("Exported "+result["path"])

    @Slot(str,object)
    def failed(self,kind,error):
        self.busy=False
        self._advance_command_id=None
        self.progress.hide()
        self.submit_button.setEnabled(self.bundle is not None and not self.analysis_current_position)
        self.step_button.setEnabled(self.bundle is not None)
        if kind=="record_analysis":
            self._analysis_record_inflight=None
            self._analysis_archive_blocked=True
        if self._shutdown_requested:
            if kind in ("record_analysis","close"):
                self._shutdown_requested=False
                self.analysis=AnalysisCoordinator()
                self.analysis_timer.start()
            else:
                self._continue_shutdown()
                return
        if kind=="analyze" and error.get("selection_hash"):
            try:
                if (error["selection_hash"]!=self.analysis_selection_hash() or self.bundle is None
                    or error.get("state_version")!=self.bundle.version):
                    return
            except ValueError:
                return
        self.pause()
        self.statusBar().showMessage(f"{kind}: {error.get('reason',error.get('code','FAILED'))}")
        self.diagnostics.appendPlainText(dumps({"action":kind,**error},pretty=True))
        if kind in ("analyze","preview","submit"):
            self.risk.setPlainText(dumps(error,pretty=True))
            self.risk_dock.raise_()

    def apply_bundle(self,bundle):
        self.analysis.gate.invalidate()
        self._analysis_retry_required=False
        changed_session=(self.bundle is not None and
                         (self.bundle.session_id,self.bundle.branch_id)!=(bundle.session_id,bundle.branch_id))
        if bundle.config and digest(bundle.config)!=digest(self.config):
            from qqq_sim.config import AppConfig
            from qqq_sim.services.calendars import load_calendar
            self.config=AppConfig(**bundle.config).validate()
            self.calendar=load_calendar(self.config.calendar_path)
        self.bundle=bundle
        if changed_session:
            self.clear_draft()
        self.risk.setPlainText("Analysis stale after state change. Select a draft or campaign to analyze.\nState "+str(bundle.version))
        selection=self.selected_entry()
        selected_key=selection.contract if selection else None
        self.chain_model.apply(bundle.snapshot.quotes,bundle.version,bundle.snapshot.snapshot_id)
        expiry=self.expiry.currentText()
        self.expiry.blockSignals(True)
        self.expiry.clear()
        self.expiry.addItems(["All observed",*sorted({x.contract.expiration.isoformat() for x in bundle.snapshot.quotes})])
        self.expiry.setCurrentText(expiry if self.expiry.findText(expiry)>=0 else "All observed")
        self.expiry.blockSignals(False)
        self.chain_proxy.set_expiry(None if self.expiry.currentText()=="All observed" else self.expiry.currentText())
        self.update_expiry_info()
        if selected_key:
            for row,entry in enumerate(self.chain_model.rows):
                if entry.contract==selected_key:
                    self.chain.selectRow(self.chain_proxy.mapFromSource(self.chain_model.index(row,0)).row())
                    break
        completed_label=(bundle.snapshot.candles[-1].label.strftime("%Y-%m-%d %H:%M")
                         if bundle.snapshot.candles else "none visible")
        self.clock_label.setText(f"{'SYNTHETIC DEMONSTRATION' if bundle.synthetic else 'HISTORICAL QQQ'}  |  Completed {completed_label}  |  Decision {bundle.decision_at:%Y-%m-%d %H:%M %Z}  |  State {bundle.version}  |  {bundle.status}")
        self.jump_time.setDateTime(QDateTime.fromString(bundle.decision_at.strftime("%Y-%m-%d %H:%M"),"yyyy-MM-dd HH:mm"))
        self.redraw_chart()
        self.account.setPlainText(dumps({"state_version":bundle.version,**bundle.account,"campaigns":bundle.campaigns},pretty=True))
        self.portfolio.setRowCount(len(bundle.positions)+len(bundle.stock_positions))
        for row,position in enumerate(bundle.positions):
            contract=position["contract"]
            label=f"{contract['expiry']} {contract['right']} {contract['strike']}"
            mark=position["mark"] if position["mark"] is not None else "— unavailable"
            for col,value in enumerate((position["campaign_id"],label,position["quantity"],mark)):
                self.portfolio.setItem(row,col,QTableWidgetItem(str(value)))
        for row,position in enumerate(bundle.stock_positions,start=len(bundle.positions)):
            mark=position["mark"] if position["mark"] is not None else "— unavailable"
            for col,value in enumerate((position["campaign_id"],"QQQ shares",position["quantity"],mark)):
                self.portfolio.setItem(row,col,QTableWidgetItem(str(value)))
        self.orders.setRowCount(len(bundle.orders)+len(bundle.stock_orders))
        for row,order in enumerate(bundle.orders):
            for col,value in enumerate((order["order_id"],order["draft"]["campaign_id"],order["status"],order["filled_units"],order["reservation"])):
                self.orders.setItem(row,col,QTableWidgetItem(str(value)))
        for row,order in enumerate(bundle.stock_orders,start=len(bundle.orders)):
            for col,value in enumerate((order["order_id"],order["campaign_id"],order["status"],abs(order["quantity"])-abs(order["remaining"]),"STOCK REDUCTION")):
                self.orders.setItem(row,col,QTableWidgetItem(str(value)))
        self.history.setPlainText("\n".join(dumps(event) for event in bundle.history[-1000:]))
        self.diagnostics.setPlainText(dumps({"state_version":bundle.version,"decision_at":bundle.decision_at,
                                           "snapshot":bundle.snapshot.snapshot_id,
                                           "quality":bundle.snapshot.diagnostics,"replay":bundle.reason,
                                           "calendar_scenario":bundle.snapshot.calendar_scenario},pretty=True))
        self.submit_button.setEnabled(not self.analysis_current_position)
        self.step_button.setEnabled(True)
        QTimer.singleShot(0,self.poll_benchmarks)

    def redraw_chart(self):
        if not self.bundle:
            return
        minutes=(1,5,15,30,60,1440)[self.aggregation.currentIndex()]
        candles=aggregate_candles(self.bundle.snapshot.candles,minutes,self.calendar)
        self.chart.apply(candles,self.bundle.decision_at,minutes)

    def update_expiry_info(self,*args):
        if self.bundle is None:
            return
        text=self.expiry.currentText()
        if not text or text=="All observed":
            self.expiry_info.setText("Select an expiration for DTE and trading cutoff")
            return
        expiration=date.fromisoformat(text)
        session=self.calendar.session(expiration)
        days=(expiration-self.bundle.decision_at.date()).days
        hours=max(0.,(session.option_close.timestamp()-self.bundle.decision_at.timestamp())/3600)
        fixing=(session.fixing_at.strftime("%Y-%m-%d %H:%M %Z") if session.fixing_at else "unavailable")
        self.expiry_info.setText(f"{days} calendar DTE · {hours:.2f} hours to trading close")
        self.expiry_info.setToolTip(f"Option trading close: {session.option_close.isoformat()}\nFixing: {fixing}\nCalendar: {session.profile_id}; scenario={session.scenario_flag}")

    def selected_entry(self):
        index=self.chain.currentIndex()
        if not index.isValid():
            return None
        source=self.chain_proxy.mapToSource(index)
        return self.chain_model.rows[source.row()] if 0<=source.row()<len(self.chain_model.rows) else None

    def add_selected(self,side):
        entry=self.selected_entry()
        if not entry:
            return
        self.draft_legs.append(OrderLeg(broker_contract(entry.contract,self.calendar),side))
        if len(self.draft_legs)==1 and self.selected_campaign is None:
            session=self.calendar.session(entry.contract.expiration)
            self.deadline.setDateTime(QDateTime.fromString((session.underlying_close-timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M"),"yyyy-MM-dd HH:mm"))
            self.cutoff.setDateTime(QDateTime.fromString(session.underlying_close.strftime("%Y-%m-%d %H:%M"),"yyyy-MM-dd HH:mm"))
        self.render_ticket()

    def render_ticket(self):
        self.ticket.setRowCount(len(self.draft_legs))
        natural=Decimal(0)
        by_key={broker_contract(e.contract,self.calendar).key:e for e in self.bundle.snapshot.quotes} if self.bundle else {}
        for row,leg in enumerate(self.draft_legs):
            c=leg.contract
            self.ticket.setItem(row,0,QTableWidgetItem(f"{c.expiry} {c.right} {c.strike}"))
            qty=QSpinBox()
            qty.setRange(-20,20)
            qty.setValue(leg.quantity)
            qty.valueChanged.connect(lambda value,row=row:self.set_quantity(row,value))
            self.ticket.setCellWidget(row,1,qty)
            self.ticket.setItem(row,2,QTableWidgetItem("BUY" if leg.quantity>0 else "SELL"))
            entry=by_key.get(c.key)
            self.ticket.setItem(row,3,QTableWidgetItem(str(round(entry.age_seconds,1)) if entry and entry.age_seconds is not None else "—"))
            if entry and entry.latest_event.valid:
                q=entry.latest_event
                natural-=leg.quantity*c.multiplier*(q.ask if leg.quantity>0 else q.bid)
        from functools import reduce
        from math import gcd
        net={}
        for leg in self.draft_legs:
            net[leg.contract.key]=net.get(leg.contract.key,0)+leg.quantity
        units=reduce(gcd,(abs(q) for q in net.values() if q),1) if not net else reduce(gcd,(abs(q) for q in net.values() if q),0) or 1
        self.limit.setValue(float(natural/units))
        self.ticket_summary.setText(f"Natural premium cash: ${natural:,.2f} total · ${natural/units:,.2f} per ratio. Fees and capital checked separately.")
        self.draft_changed()

    def set_quantity(self,row,value):
        if not value:
            self.draft_legs.pop(row)
        else:
            self.draft_legs[row]=OrderLeg(self.draft_legs[row].contract,value)
        self.render_ticket()

    def undo_leg(self):
        if self.draft_legs:
            self.draft_legs.pop()
            self.render_ticket()

    def clear_draft(self,reset_campaign=True):
        self.draft_legs=[]
        self.ticket.setRowCount(0)
        self.ticket_summary.setText("Select exact contracts from the observed chain.")
        if reset_campaign:
            self.selected_campaign=None
            self.campaign_edit.setText("campaign-"+uuid4().hex[:8])
            self.intent.setCurrentText("OPEN")
        self.draft_changed()

    def draft_changed(self,*args):
        self.analysis_current_position=False
        self._analysis_retry_required=False
        self.submit_button.setEnabled(self.bundle is not None and not self.busy)
        self.selection_version+=1
        self.analysis.gate.invalidate()
        if hasattr(self,"debounce"):
            self.debounce.start()

    def make_draft(self):
        if not self.draft_legs:
            raise ValueError("EMPTY_ORDER: select contracts first")
        return OrderDraft(self.campaign_edit.text().strip(),tuple(self.draft_legs),Decimal(str(self.limit.value())),
                          editor_time(self.deadline),editor_time(self.cutoff),order_type=self.order_type.currentText(),
                          max_total_fees=Decimal(str(self.fee_cap.value())),tif=self.tif.currentText(),intent=self.intent.currentText())

    def draft_action(self,kind):
        try:
            if self.analysis_current_position:
                raise ValueError("CURRENT_POSITION_VIEW: choose Close or Adjust / roll to create a trade proposal")
            self.send(kind,draft=self.make_draft())
        except Exception as exc:
            from qqq_sim.cli import error_record
            self.failed(kind,error_record(exc))

    def analyze(self):
        if self.busy or not self.draft_legs or not self.bundle:
            return
        try:
            self.debounce.stop()
            self._analysis_archive_blocked=False
            if self._analysis_retry_required:
                self.selection_version+=1
                self.analysis.gate.invalidate()
                self._analysis_retry_required=False
            draft=self.make_draft()
            self.send("analyze",draft=draft,selection_hash=self.analysis_selection_hash(),
                      current_position=self.analysis_current_position)
        except Exception as exc:
            from qqq_sim.cli import error_record
            self.failed("analyze",error_record(exc))

    def analysis_selection_hash(self):
        return digest({"draft":self.make_draft(),"selection_version":self.selection_version,
                       "current_position":self.analysis_current_position})

    def _has_analysis_request(self,request_hash):
        return bool(request_hash and (any(item["request_hash"]==request_hash for item in self._analysis_ready)
                    or any(record[1]==request_hash for record in self._analysis_futures.values())))

    def _queue_shutdown_analysis(self,identity,request_hash,archive_path):
        if request_hash and not self._has_analysis_request(request_hash):
            self._analysis_ready.append({"identity":identity,"request_hash":request_hash,"archive_path":archive_path,
                "result":{"status":"CANCELED","reason":"APPLICATION_SHUTDOWN",
                          "hold_probability":None,"managed_probability":None}})

    def _collect_analysis_results(self,terminalize=False):
        for future,(identity,request_hash,archive_path) in list(self._analysis_futures.items()):
            if not future.done() and not terminalize:
                continue
            del self._analysis_futures[future]
            # Keep the completed coordinator future until another identity is
            # requested: a duplicate prepare reply may precede its archive ACK.
            try:
                if not future.done() or future.cancelled():
                    result={"status":"CANCELED","reason":"APPLICATION_SHUTDOWN" if terminalize else "SUPERSEDED_BEFORE_ANALYSIS",
                            "hold_probability":None,"managed_probability":None}
                else:
                    returned_identity,result=future.result()
                    if returned_identity!=identity:
                        raise ValueError("ANALYSIS_IDENTITY_MISMATCH")
            except Exception as exc:
                from qqq_sim.cli import error_record
                result={**error_record(exc),"hold_probability":None,"managed_probability":None}
            if request_hash:
                if not self._has_analysis_request(request_hash):
                    self._analysis_ready.append({"identity":identity,"request_hash":request_hash,"archive_path":archive_path,"result":result})
            if self.analysis.gate.accepts(identity) and not self._shutdown_requested:
                self._analysis_retry_required=result.get("status")=="FAILED"
                self.risk.setPlainText(dumps(result,pretty=True))
                self.risk_dock.raise_()
    def _flush_analysis_record(self):
        if self._analysis_ready and not self.busy and not self._analysis_archive_blocked:
            self._analysis_record_inflight=self._analysis_ready[0]
            self.send("record_analysis",**self._analysis_record_inflight)
            return True
        return False

    def poll_analysis(self):
        self._collect_analysis_results()
        if not self._shutdown_requested:
            self._flush_analysis_record()
        self.poll_benchmarks()

    def _continue_shutdown(self):
        if self.busy:
            return
        if not self._flush_analysis_record():
            self.send("close")

    def poll_benchmarks(self):
        if self._shutdown_requested or self.bundle is None:
            return
        current=(self.bundle.session_id,self.bundle.branch_id)
        for key,future in list(self._benchmark_futures.items()):
            if not future.done():
                continue
            del self._benchmark_futures[key]
            if key[:2]!=current:
                continue
            try:
                result=future.result()
                self._benchmark_ready.append((key,result))
            except Exception as exc:
                from qqq_sim.cli import error_record
                self.diagnostics.appendPlainText(dumps({"action":"initial_benchmark",**error_record(exc)},pretty=True))
                if self._benchmark_retries.get(key,0)<1:
                    self._benchmark_retries[key]=self._benchmark_retries.get(key,0)+1
                    self._benchmark_requested.discard(key)
        if self.busy:
            return
        while self._benchmark_ready:
            key,result=self._benchmark_ready.pop(0)
            if key[:2]==current:
                self.send("record_benchmark",campaign_id=result["campaign_id"],benchmark_id=result["benchmark_id"],result=result)
                return
        for pending in getattr(self.bundle,"pending_benchmarks",()):
            key=(*current,pending["benchmark_id"])
            if key not in self._benchmark_requested:
                self._benchmark_requested.add(key)
                self.send("prepare_benchmark",campaign_id=pending["campaign_id"],benchmark_id=pending["benchmark_id"])
                return

    def step(self):
        if self.busy or not self.bundle:
            return
        if self.step_unit.currentIndex()==0:
            self.send("advance",minutes=self.step_count.value())
        else:
            from qqq_sim.replay.calendar import add_elapsed_seconds
            self.send("advance",target=add_elapsed_seconds(self.bundle.decision_at,self.step_count.value()*60))

    def toggle_play(self):
        if self.play_timer.isActive():
            self.pause()
        else:
            self.play_timer.start()
            self.play_button.setText("Pause")

    def pause(self):
        self.play_timer.stop()
        self.play_button.setText("Play")
        if self.worker.controller:
            self.worker.controller.cancel_requested.set()
        if self.busy and self._advance_command_id:
            self.statusBar().showMessage("Stopping after the current minute; the last saved state will be displayed.")

    def next_session(self):
        if not self.bundle:
            return
        day=self.bundle.decision_at.date()+timedelta(days=1)
        while self.calendar.session(day).status!="OPEN":
            day+=timedelta(days=1)
        self.send("advance",target=self.calendar.session(day).exchange_open)

    def select_campaign(self,action):
        if not self.bundle or not self.bundle.campaigns:
            return
        ids=[c["campaign_id"] for c in self.bundle.campaigns]
        campaign_id=ids[0]
        if len(ids)>1:
            campaign_id,ok=QInputDialog.getItem(self,"Select campaign","Campaign",ids,editable=False)
            if not ok:
                return
        campaign=next(c for c in self.bundle.campaigns if c["campaign_id"]==campaign_id)
        self.selected_campaign=campaign_id
        self.campaign_edit.setText(campaign_id)
        self.draft_legs=[]
        for p in self.bundle.positions:
            if p["campaign_id"]!=campaign_id:
                continue
            sign=1 if action=="ANALYZE" else -1
            self.draft_legs.append(OrderLeg(Contract.from_dict(p["contract"]),sign*p["quantity"]))
        self.intent.setCurrentText("CLOSE" if action=="CLOSE" else "ROLL" if action=="ROLL" else "ADJUST")
        for editor,key in ((self.deadline,"deadline"),(self.cutoff,"hard_cutoff")):
            editor.setDateTime(QDateTime.fromString(datetime.fromisoformat(campaign[key]).strftime("%Y-%m-%d %H:%M"),"yyyy-MM-dd HH:mm"))
        self.render_ticket()
        self.ticket_dock.raise_()
        if action=="ANALYZE":
            self.analysis_current_position=True
            self.submit_button.setEnabled(False)
            self.analyze()

    def cancel_order(self):
        row=self.orders.currentRow()
        if self.bundle:
            orders=self.bundle.orders+self.bundle.stock_orders
            if 0<=row<len(orders):
                self.send("cancel",order_id=orders[row]["order_id"])

    def reduce_stock(self):
        if not self.bundle or not self.bundle.stock_positions:
            self.show_info("No assigned stock is held. Stock reduction orders require later explicit stock bid/ask observations.")
            return
        positions=self.bundle.stock_positions
        labels=[f"{p['campaign_id']} · {p['quantity']} shares" for p in positions]
        selection,ok=QInputDialog.getItem(self,"Reduce stock","Position",labels,editable=False)
        if not ok:
            return
        position=positions[labels.index(selection)]
        amount,ok=QInputDialog.getInt(self,"Reduce stock","Number of shares to reduce",abs(position["quantity"]),1,abs(position["quantity"]))
        if not ok:
            return
        limit,ok=QInputDialog.getDouble(self,"Stock limit","Minimum signed cash per share ($): positive for a sale, negative for a purchase",0,-100000,100000,2)
        if ok:
            self.send("stock_reduce",campaign_id=position["campaign_id"],quantity=-amount if position["quantity"]>0 else amount,minimum_cash_per_share=str(limit))

    def restore_session(self):
        path,_=QFileDialog.getOpenFileName(self,"Restore a local session",str(writable_path("runtime/sessions")),"SQLite sessions (*.sqlite)")
        if path:
            self.send("restore",path=path)

    def export_session(self):
        path,_=QFileDialog.getSaveFileName(self,"Export frozen session",str(writable_path("runtime/results/session-export.json")),"JSON (*.json)")
        if path:
            self.send("export",path=path)

    def branch(self):
        sequence,ok=QInputDialog.getInt(self,"Research branch","Committed event sequence (0 = current)",0,0,2**30)
        if ok:
            self.send("branch",sequence=sequence or None,reason="User research branch; future path may already have been inspected")

    def load_pack(self):
        path=QFileDialog.getExistingDirectory(self,"Load model-pack directory",str(writable_path("runtime/model_packs")))
        if path:
            self.send("load_pack",path=path)

    def load_parameters(self):
        path,_=QFileDialog.getOpenFileName(self,"Load known-as-of rate / carry / path-IV / fixing inputs",str(writable_path("runtime")),"JSON (*.json)")
        if path:
            self.send("load_parameters",path=path)

    def show_info(self,message):
        self.diagnostics.setPlainText(message)
        self.statusBar().showMessage(message.splitlines()[0])

    def _restore_layout(self):
        path=writable_path(Path(self.config.runtime_root)/"config/window.json")
        if path.exists():
            try:
                record=json.loads(path.read_text(encoding="utf-8"))
                self.restoreGeometry(QByteArray.fromHex(record["geometry"].encode()))
                self.restoreState(QByteArray.fromHex(record["state"].encode()))
            except (ValueError,KeyError):
                pass

    def closeEvent(self,event):
        if self._closing:
            event.accept()
            return
        event.ignore()
        if self._shutdown_requested:
            return
        self._shutdown_requested=True
        self.pause()
        self.debounce.stop()
        self.analysis_timer.stop()
        self.analysis.close()
        self._collect_analysis_results(terminalize=True)
        self._analysis_archive_blocked=False
        try:
            save_json(Path(self.config.runtime_root)/"config/window.json",{"geometry":bytes(self.saveGeometry().toHex()).decode(),"state":bytes(self.saveState().toHex()).decode()})
        except Exception as exc:
            from qqq_sim.cli import error_record
            self.failed("close",error_record(exc))
            return
        self._continue_shutdown()
