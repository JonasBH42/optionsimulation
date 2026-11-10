from qqq_sim.data.cache import ByteLRU


def assert_accounting(cache):
    assert cache.bytes == sum(size for _, size in cache._items.values())
    assert cache._transient_bytes == sum(cache._items[key][1] for key in cache._transient)
    assert cache.bytes <= cache.max_bytes


def test_transient_observations_never_evict_source_indexes():
    cache = ByteLRU(100)
    cache.put("index", "retained", 70)
    cache.put("quote1", "first", 20, transient=True)
    cache.put("quote2", "second", 20, transient=True)
    assert cache.get("index") == "retained"
    assert cache.get("quote1") is None
    assert cache.get("quote2") == "second"
    assert cache.put("too-large-quote", "uncached", 40, transient=True) == "uncached"
    assert cache.get("too-large-quote") is None
    assert cache.get("quote2") == "second"
    assert_accounting(cache)


def test_retained_write_evicts_transient_before_older_source_index():
    cache = ByteLRU(100)
    cache.put("old-index", "old", 40)
    cache.put("quote", "disposable", 40, transient=True)
    cache.put("new-index", "new", 40)
    assert cache.get("quote") is None
    assert cache.get("old-index") == "old"
    assert cache.get("new-index") == "new"
    assert_accounting(cache)


def test_transient_and_retained_entries_each_respect_lru_reads():
    cache = ByteLRU(100)
    cache.put("index1", "one", 30)
    cache.put("index2", "two", 30)
    cache.put("quote1", "first", 20, transient=True)
    cache.put("quote2", "second", 20, transient=True)
    cache.get("quote1")
    cache.put("quote3", "third", 20, transient=True)
    assert cache.get("quote2") is None
    cache.get("index1")
    cache.put("index3", "three", 60)
    assert cache.get("quote1") is None and cache.get("quote3") is None
    assert cache.get("index2") is None
    assert cache.get("index1") == "one" and cache.get("index3") == "three"
    assert_accounting(cache)


def test_replacement_class_changes_oversized_put_and_clear_keep_exact_accounting():
    cache = ByteLRU(100)
    cache.put("same", "quote", 20, transient=True)
    cache.put("same", "index", 50)
    assert cache.get("same") == "index"
    assert "same" not in cache._transient
    assert_accounting(cache)
    cache.put("same", "new-quote", 30, transient=True)
    assert cache.get("same") == "new-quote"
    assert list(cache._transient) == ["same"]
    assert_accounting(cache)
    cache.put("same", "too-big", 101)
    assert cache.get("same") is None
    assert_accounting(cache)
    cache.put("final-index", "index", 50)
    cache.put("final-quote", "quote", 20, transient=True)
    cache.clear()
    assert not cache._items and not cache._transient
    assert cache.bytes == cache._transient_bytes == 0


def test_zero_size_transient_is_admitted_without_displacing_full_index():
    cache = ByteLRU(100)
    cache.put("index", "index", 100)
    cache.put("quote", "empty", 0, transient=True)
    assert cache.get("index") == "index"
    assert cache.get("quote") == "empty"
    assert_accounting(cache)
