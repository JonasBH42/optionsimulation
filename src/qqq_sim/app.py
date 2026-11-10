from __future__ import annotations
import sys


def configure_fonts(app):
    """Windows offscreen Qt needs an explicit font source for render tests."""
    from pathlib import Path
    from PySide6.QtGui import QFont, QFontDatabase
    font=Path(r"C:\Windows\Fonts\segoeui.ttf")
    if font.is_file():
        QFontDatabase.addApplicationFont(str(font))
        app.setFont(QFont("Segoe UI",9))


def launch(config, at=None, synthetic=False):
    from .services.startup import startup_report,setup_guidance
    report=startup_report(config)
    try:
        from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QPlainTextEdit,QPushButton,QDockWidget
        from PySide6.QtCore import Qt
    except ImportError:
        from .serialization import dumps
        print(dumps(report,pretty=True),file=sys.stderr)
        return 2
    app=QApplication.instance() or QApplication(sys.argv[:1])
    configure_fonts(app)
    app.setApplicationName("QQQ Historical Simulator")
    app.setOrganizationName("Local QQQ Simulation")
    if not report["runtime_ready"]:
        # A plain nonmodal setup window needs no session or writable UI state.
        window=QWidget()
        window.setWindowTitle("QQQ Simulator: setup required")
        layout=QVBoxLayout(window)
        message=QPlainTextEdit(setup_guidance(report));message.setReadOnly(True)
        layout.addWidget(message)
        close=QPushButton("Close");close.clicked.connect(window.close);layout.addWidget(close)
        window.resize(750,430);window.show()
        app.exec()
        return 2
    from .ui.main_window import MainWindow
    window=MainWindow(config,at,synthetic=synthetic,auto_open=report["can_open_session"])
    window.startup_report=report
    if report["issues"]:
        dock=QDockWidget("Startup checks",window)
        dock.setObjectName("Startup checks")
        message=QPlainTextEdit(setup_guidance(report));message.setReadOnly(True)
        dock.setWidget(message)
        window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,dock)
    window.show()
    return app.exec()
