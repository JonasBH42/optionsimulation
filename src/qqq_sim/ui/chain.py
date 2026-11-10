from __future__ import annotations
from decimal import Decimal
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QSortFilterProxyModel
from PySide6.QtGui import QColor

CONTRACT_ROLE = int(Qt.ItemDataRole.UserRole) + 1


class ChainTableModel(QAbstractTableModel):
    HEADERS = ("Expiry", "Right", "Strike ($)", "Bid ($)", "Ask ($)", "Mid ($)", "Last trade ($)",
               "IV (%)", "Delta · vendor", "Gamma · vendor", "Theta · vendor", "Vega · vendor",
               "Bid size", "Ask size", "Volume", "OI · known", "Age (s)", "Quality")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = ()
        self.state_version = -1
        self.snapshot_id = ""

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def values(self, row):
        item = self.rows[row]
        quote = item.quote
        greeks = dict(quote.vendor_greeks) if quote else {}
        return (item.contract.expiration.isoformat(), item.contract.right,
                item.contract.strike, quote.bid if quote else None,
                quote.ask if quote else None, quote.mid if quote else None,item.last_trade_price,
                quote.iv * 100 if quote and quote.iv is not None else None,
                *(greeks.get(k) for k in ("delta", "gamma", "theta", "vega")),
                quote.bid_size if quote else None, quote.ask_size if quote else None,
                item.volume, item.known_oi, item.age_seconds, item.status)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.rows) or not 0 <= index.column() < len(self.HEADERS):
            return None
        row = self.rows[index.row()]
        value = self.values(index.row())[index.column()]
        if role == Qt.ItemDataRole.UserRole:
            return float(value) if isinstance(value, Decimal) else value
        if role == CONTRACT_ROLE:
            return row.contract
        if role == Qt.ItemDataRole.ToolTipRole:
            event = row.latest_event
            if index.column()==6:
                return (f"Recorded trade: {row.last_trade_at.isoformat() if row.last_trade_at else 'unavailable'}\n"
                        f"Source: {row.last_trade_source or 'unavailable'}\nQuality: {', '.join(row.last_trade_quality) or 'no additional flags'}\n"
                        "Simulated fills use eligible bid/ask events, separately from last-trade display.")
            return (f"{row.contract.contract_id}\nSource: {event.source_id}\n"
                    f"Event: {event.event_at.isoformat()}\nAvailable: {event.available_at.isoformat()}\n"
                    f"Quality: {row.status}; {', '.join(event.quality_flags)}\n"
                    f"Universe: {row.universe_basis}\nState {self.state_version}, snapshot {self.snapshot_id}")
        if role == Qt.ItemDataRole.ForegroundRole and (value is None or row.status not in ("OK", "CURRENT") or index.column()==6 and row.last_trade_quality):
            return QColor("#a56820")
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (Decimal, float, int)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.DisplayRole:
            if value is None:
                return "— unavailable"
            if isinstance(value, (float, Decimal)):
                return f"{value:,.4f}" if index.column() in (8, 9, 10, 11) else f"{value:,.2f}"
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal and 0 <= section < len(self.HEADERS):
            return self.HEADERS[section]
        return None

    def flags(self, index):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable if index.isValid() else Qt.ItemFlag.NoItemFlags

    def apply(self, rows, state_version, snapshot_id):
        rows = tuple(rows)
        same_keys = tuple(x.contract for x in rows) == tuple(x.contract for x in self.rows)
        self.state_version, self.snapshot_id = state_version, snapshot_id
        if same_keys:
            self.rows = rows
            if rows:
                self.dataChanged.emit(self.index(0, 0), self.index(len(rows)-1, len(self.HEADERS)-1),
                                      [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.UserRole, Qt.ItemDataRole.ToolTipRole,
                                       Qt.ItemDataRole.ForegroundRole])
        else:
            self.beginResetModel()
            self.rows = rows
            self.endResetModel()


class ChainProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.expiry = None
        self.setSortRole(Qt.ItemDataRole.UserRole)

    def set_expiry(self, expiry):
        self.beginFilterChange()
        self.expiry = expiry or None
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(self, row, parent):
        return self.expiry is None or self.sourceModel().rows[row].contract.expiration.isoformat() == self.expiry

    def lessThan(self, left, right):
        a, b = left.data(Qt.ItemDataRole.UserRole), right.data(Qt.ItemDataRole.UserRole)
        descending = self.sortOrder() == Qt.SortOrder.DescendingOrder
        if a is None:
            return b is not None and descending
        if b is None:
            return not descending
        return a < b
