import json
import pytest


@pytest.fixture
def module():
    from qqq_sim.risk.offline import hit_oracle
    return hit_oracle


def sharing_error(code):
    error=PermissionError("Simulated Windows file sharing error")
    error.winerror=code
    return error


@pytest.mark.parametrize("code",[5,32,33])
def test_transient_windows_sharing_retry_keeps_old_commit_until_success(tmp_path,monkeypatch,module,code):
    target=tmp_path/"state.json"
    old=b'{"old":true}'
    target.write_bytes(old)
    original=module.os.replace
    calls=[];sleeps=[]
    def interrupted(source,destination):
        calls.append((source,destination))
        assert target.read_bytes()==old
        if len(calls)<=2:
            raise sharing_error(code)
        return original(source,destination)
    monkeypatch.setattr(module.os,"replace",interrupted)
    monkeypatch.setattr(module.time,"sleep",sleeps.append)
    module.atomic_json(target,{"new":True})
    assert json.loads(target.read_text())=={"new":True}
    assert len(calls)==3
    assert sleeps==[.01,.02]
    assert not target.with_suffix(".json.tmp").exists()


def test_permanent_sharing_error_propagates_after_bounded_attempts_and_cleans_temp(tmp_path,monkeypatch,module):
    target=tmp_path/"state.json"
    old=b'{"old":true}'
    target.write_bytes(old)
    error=sharing_error(5)
    calls=[];sleeps=[]
    def failing(*args):
        calls.append(args)
        raise error
    monkeypatch.setattr(module.os,"replace",failing)
    monkeypatch.setattr(module.time,"sleep",sleeps.append)
    with pytest.raises(PermissionError) as caught:
        module.atomic_json(target,{"new":True})
    assert caught.value is error
    assert len(calls)==6
    assert sleeps==[.01,.02,.04,.08,.16]
    assert target.read_bytes()==old
    assert not target.with_suffix(".json.tmp").exists()


def test_other_io_error_is_not_retried(tmp_path,monkeypatch,module):
    target=tmp_path/"state.json"
    target.write_text('{"old":true}')
    error=OSError("Simulated disk-full error")
    error.winerror=112
    calls=[]
    def failing(*args):
        calls.append(args)
        raise error
    monkeypatch.setattr(module.os,"replace",failing)
    monkeypatch.setattr(module.time,"sleep",lambda value:pytest.fail("Non-sharing errors must not retry"))
    with pytest.raises(OSError) as caught:
        module.atomic_json(target,{"new":True})
    assert caught.value is error
    assert len(calls)==1
    assert json.loads(target.read_text())=={"old":True}
    assert not target.with_suffix(".json.tmp").exists()


def test_serialization_error_preserves_commit_and_cleans_partial_temp(tmp_path,module):
    target=tmp_path/"state.json"
    old=b'{"old":true}'
    target.write_bytes(old)
    with pytest.raises(TypeError):
        module.atomic_json(target,{"bad":object()})
    assert target.read_bytes()==old
    assert not target.with_suffix(".json.tmp").exists()
