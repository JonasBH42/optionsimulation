from dataclasses import replace

from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QPushButton, QScrollArea

from qqq_sim.config import AppConfig
from qqq_sim.ui.main_window import MainWindow


def test_small_work_area_keeps_frame_and_scrolled_action_controls_reachable(qtbot,tmp_path):
    window=MainWindow(replace(AppConfig(),runtime_root=str(tmp_path)),auto_open=False)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(30)
    try:
        available=QRect(0,0,790,590)
        window.fit_to_screen(available)
        qtbot.wait(30)
        assert available.contains(window.frameGeometry())
        scroll=window.ticket_scroll
        assert isinstance(scroll,QScrollArea)
        window.ticket_dock.raise_()
        scroll.ensureWidgetVisible(window.submit_button,8,8)
        qtbot.wait(10)
        button=QRect(window.submit_button.mapTo(scroll.viewport(),QPoint()),window.submit_button.size())
        assert scroll.viewport().rect().contains(button)
        assert scroll.verticalScrollBar().maximum()>0
        scroll.ensureWidgetVisible(window.campaign_edit,8,8)
        qtbot.wait(10)
        field=QRect(window.campaign_edit.mapTo(scroll.viewport(),QPoint()),window.campaign_edit.size())
        assert scroll.viewport().rect().contains(field)
        window.portfolio_dock.raise_()
        portfolio_scroll=window.portfolio_dock.widget()
        assert isinstance(portfolio_scroll,QScrollArea)
        cancel=next(button for button in portfolio_scroll.findChildren(QPushButton)
                    if button.text()=="Cancel selected working order")
        portfolio_scroll.ensureWidgetVisible(cancel,8,8)
        qtbot.wait(10)
        button=QRect(cancel.mapTo(portfolio_scroll.viewport(),QPoint()),cancel.size())
        assert portfolio_scroll.viewport().rect().contains(button)
        for dock,editor in ((window.account_dock,window.account),(window.risk_dock,window.risk)):
            dock.raise_()
            editor.setPlainText("\n".join("Readable record "+str(i) for i in range(100)))
            qtbot.wait(10)
            editor.verticalScrollBar().setValue(editor.verticalScrollBar().maximum())
            assert editor.verticalScrollBar().maximum()>0
            assert available.contains(window.frameGeometry())
    finally:
        window.close()
        qtbot.waitUntil(lambda:not window.thread.isRunning(),timeout=10000)


def test_restored_oversized_layout_and_smaller_screen_are_constrained(qtbot,tmp_path,monkeypatch):
    def oversized(self):
        self.resize(5000,4000)
        self.move(9000,9000)
    monkeypatch.setattr(MainWindow,"_restore_layout",oversized)
    window=MainWindow(replace(AppConfig(),runtime_root=str(tmp_path)),auto_open=False)
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(30)
    try:
        assert window.screen().availableGeometry().contains(window.frameGeometry())
        smaller=QRect(10,20,780,580)
        window.fit_to_screen(smaller)
        qtbot.wait(10)
        assert smaller.contains(window.frameGeometry())
        assert window.maximumWidth()<=smaller.width()
        assert window.maximumHeight()<=smaller.height()
    finally:
        window.close()
        qtbot.waitUntil(lambda:not window.thread.isRunning(),timeout=10000)
