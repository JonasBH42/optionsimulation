from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QPointF, Qt
from PySide6.QtGui import QPicture, QPainter
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

NY = ZoneInfo("America/New_York")


class NYAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(v, NY).strftime("%m-%d\n%H:%M") for v in values]


class CandlestickItem(pg.GraphicsObject):
    """Chunked pictures contain only already-visible candles (APP-C10)."""
    def __init__(self):
        super().__init__()
        self.chunks = []
        self.signatures = []
        self.bounds = QRectF()
        self.candles = ()

    def set_candles(self, candles, minutes=1):
        self.prepareGeometryChange()
        self.candles = tuple(candles)
        half = minutes * 60 * .35
        pictures, signatures = [], []
        for start in range(0, len(self.candles), 256):
            chunk = self.candles[start:start+256]
            sig = (minutes, tuple((c.label, c.open, c.high, c.low, c.close, c.partial) for c in chunk))
            index = start // 256
            if index < len(self.signatures) and self.signatures[index] == sig:
                picture = self.chunks[index]
            else:
                picture = QPicture()
                painter = QPainter(picture)
                for c in chunk:
                    x = c.label.timestamp()
                    color = "#168879" if c.close >= c.open else "#ce574c"
                    painter.setPen(pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine if c.partial else Qt.PenStyle.SolidLine))
                    painter.setBrush(pg.mkBrush(color))
                    painter.drawLine(QPointF(x, c.low), QPointF(x, c.high))
                    if c.open == c.close:
                        painter.drawLine(QPointF(x-half, c.open), QPointF(x+half, c.open))
                    else:
                        painter.drawRect(QRectF(x-half, min(c.open,c.close), 2*half, abs(c.close-c.open)))
                painter.end()
            pictures.append(picture)
            signatures.append(sig)
        self.chunks, self.signatures = pictures, signatures
        if self.candles:
            xs = [c.label.timestamp() for c in self.candles]
            lo, hi = min(c.low for c in self.candles), max(c.high for c in self.candles)
            self.bounds = QRectF(min(xs)-half, lo, max(xs)-min(xs)+2*half, max(hi-lo, .001))
        else:
            self.bounds = QRectF()
        self.update()

    def paint(self, painter, option, widget=None):
        for chunk in self.chunks:
            painter.drawPicture(0, 0, chunk)

    def boundingRect(self):
        return self.bounds


class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.caption = QLabel("No completed candle selected")
        layout.addWidget(self.caption)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("#fbfcfe")
        layout.addWidget(self.graphics)
        self.price = self.graphics.addPlot(row=0, col=0, axisItems={"bottom": NYAxis(orientation="bottom")})
        self.volume = self.graphics.addPlot(row=1, col=0, axisItems={"bottom": NYAxis(orientation="bottom")})
        self.graphics.ci.layout.setRowStretchFactor(0, 4)
        self.graphics.ci.layout.setRowStretchFactor(1, 1)
        self.price.setLabel("left", "QQQ", units="USD")
        self.price.getAxis("bottom").setHeight(42)
        self.volume.getAxis("bottom").setHeight(42)
        self.volume.setLabel("left", "Volume")
        self.volume.setXLink(self.price)
        self.price.showGrid(x=True, y=True, alpha=.12)
        self.item = CandlestickItem()
        self.price.addItem(self.item)
        self.volume_bars = pg.BarGraphItem(x=[], height=[], width=42, brush="#9aafc5", pen=None)
        self.volume.addItem(self.volume_bars)
        self.boundary = pg.InfiniteLine(angle=90, pen=pg.mkPen("#4366b0", style=Qt.PenStyle.DashLine))
        self.price.addItem(self.boundary)
        self.crosshair = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#a2abb8"))
        self.price.addItem(self.crosshair, ignoreBounds=True)
        self.graphics.scene().sigMouseMoved.connect(self.hover)
        self.visible = ()

    def apply(self, candles, decision_at, minutes=1):
        self.visible = tuple(candles)
        self.item.set_candles(self.visible, minutes)
        self.volume_bars.setOpts(x=np.array([c.label.timestamp() for c in candles]),
                                 height=np.array([c.volume for c in candles]), width=minutes*42)
        self.boundary.setValue(decision_at.timestamp())
        self.caption.setText(f"Historical QQQ · decision {decision_at.astimezone(NY):%Y-%m-%d %H:%M:%S %Z} · {len(candles)} visible bars")
        if candles:
            self.price.setXRange(candles[max(0,len(candles)-100)].label.timestamp()-minutes*60,
                                 decision_at.timestamp()+minutes*180, padding=0)

    def hover(self, point):
        if not self.visible or not self.price.sceneBoundingRect().contains(point):
            return
        x = self.price.vb.mapSceneToView(point).x()
        c = min(self.visible, key=lambda c: abs(c.label.timestamp()-x))
        self.crosshair.setValue(c.label.timestamp())
        self.caption.setText(f"{c.label.astimezone(NY):%Y-%m-%d %H:%M %Z} · O {c.open:.2f} H {c.high:.2f} L {c.low:.2f} C {c.close:.2f} · V {c.volume:,.0f}" + (" · PARTIAL" if c.partial else ""))
