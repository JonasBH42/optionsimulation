from collections import OrderedDict
from threading import RLock


class ByteLRU:
    """Immutable values only; explicit conservative byte-budget eviction."""
    def __init__(self, max_bytes: int = 256 * 1024 * 1024):
        self.max_bytes, self.bytes = max_bytes, 0
        self._items, self._lock = OrderedDict(), RLock()
        self._transient, self._transient_bytes = OrderedDict(), 0

    def get(self, key):
        with self._lock:
            if key not in self._items:
                return None
            value, size = self._items.pop(key)
            self._items[key] = value, size
            if key in self._transient:
                self._transient.move_to_end(key)
            return value

    def _discard(self, key):
        _, size = self._items.pop(key)
        self.bytes -= size
        if key in self._transient:
            del self._transient[key]
            self._transient_bytes -= size

    def put(self, key, value, size: int, *, transient=False):
        """Keep raw/index summaries ahead of disposable canonical observations.

        Transient writes may replace other transient entries, and otherwise
        decline admission. They never evict retained source/index state.
        """
        with self._lock:
            if key in self._items:
                self._discard(key)
            if size > self.max_bytes:
                return value
            if transient and self.bytes - self._transient_bytes + size > self.max_bytes:
                return value
            while self._items and self.bytes + size > self.max_bytes:
                key_to_evict = next(iter(self._transient)) if self._transient else next(iter(self._items))
                self._discard(key_to_evict)
            self._items[key] = value, size
            self.bytes += size
            if transient:
                self._transient[key] = None
                self._transient_bytes += size
            return value

    def clear(self):
        with self._lock:
            self._items.clear()
            self._transient.clear()
            self.bytes = 0
            self._transient_bytes = 0
