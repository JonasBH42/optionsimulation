"""Immutable analytic inputs/results, separate from economic ledger versions."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import json
from qqq_sim.broker.models import BrokerError
from qqq_sim.data.cache import ByteLRU
from qqq_sim.serialization import primitive,digest,dumps
from .analysis import AnalysisIdentity

MAX_RECORD_BYTES=64*1024*1024


def _identity(value):
    return value if isinstance(value,AnalysisIdentity) else AnalysisIdentity(**value)


def _cache(controller):
    cache=getattr(controller,"_analytic_record_cache",None)
    if cache is None:
        cache=ByteLRU(min(16*1024*1024,max(1024*1024,controller.config.cache_bytes//8)))
        controller._analytic_record_cache=cache
    return cache


def register_analysis_request(controller,prepared,*,draft=None,current_position=False,request_context=None):
    c=controller
    identity=_identity(prepared["identity"])
    if (identity.session_id,identity.branch_id,identity.state_version,identity.snapshot_hash)!=(c.session_id,c.branch_id,c.version,c.snapshot.snapshot_id):
        raise BrokerError("ANALYSIS_REQUEST_STALE_AT_REGISTRATION")
    request=primitive({key:value for key,value in prepared.items() if key not in ("identity","request_hash","payload_hash","snapshot_payload_hash","archive_path","cached_result")})
    payload={"request":request,"draft":primitive(draft),"current_position":bool(current_position),"request_context":primitive(request_context or {})}
    snapshot=primitive(c.snapshot)
    encoded=dumps({"payload":payload,"snapshot":snapshot})
    if len(encoded.encode("utf-8"))>MAX_RECORD_BYTES:
        raise BrokerError("ANALYSIS_REQUEST_RESOURCE_LIMIT",maximum_bytes=MAX_RECORD_BYTES)
    request_hash=digest(identity)
    stored=c.store.register_analysis_request(request_hash,identity,payload,snapshot)
    _cache(c).put(("request",request_hash),encoded,len(encoded.encode("utf-8")))
    c._latest_analysis_request_hash=request_hash
    result=c.store.analysis_result(request_hash)
    response={**json.loads(dumps(request)),"identity":identity,"request_hash":request_hash,"payload_hash":stored["payload_hash"],
              "snapshot_payload_hash":stored["snapshot_payload_hash"],"archive_path":str(c.store.path)}
    if result is not None:
        response["cached_result"]=result["payload"]["result"]
    return response


def _record(store,identity,result,request_hash=None,controller=None):
    identity=_identity(identity)
    request_hash=request_hash or digest(identity)
    request=store.analysis_request(request_hash)
    if primitive(identity)!=request["identity"]:
        raise BrokerError("ANALYSIS_RESULT_IDENTITY_MISMATCH",request_hash=request_hash)
    result=primitive(result)
    encoded=dumps(result)
    if len(encoded.encode("utf-8"))>MAX_RECORD_BYTES:
        raise BrokerError("ANALYSIS_RESULT_RESOURCE_LIMIT",maximum_bytes=MAX_RECORD_BYTES)
    result_hash=digest(result)
    old=store.analysis_result(request_hash)
    if old is not None:
        if old["payload"]["result_hash"]!=result_hash:
            raise BrokerError("ANALYSIS_RESULT_CONFLICT",request_hash=request_hash)
        record=old
    else:
        payload={"schema_version":1,"request_hash":request_hash,"identity":request["identity"],"request_payload_hash":request["payload_hash"],
                 "snapshot_payload_hash":request["snapshot_payload_hash"],"result":result,"result_hash":result_hash,
                 "recorded_at":datetime.now(timezone.utc).isoformat(),"purpose":"FROZEN_ANALYTIC_OBSERVATION_NO_ECONOMIC_ACTION"}
        record=store.record_analysis(request_hash,identity.branch_id,identity.state_version,identity.selection_hash,identity.pack_id,
                                     str(result.get("status","UNAVAILABLE")),payload)
    current=controller is not None and (identity.session_id,identity.branch_id,identity.state_version,identity.snapshot_hash,identity.pack_id)==(
        controller.session_id,controller.branch_id,controller.version,controller.snapshot.snapshot_id,controller._pack_hash)
    selected=current and getattr(controller,"_latest_analysis_request_hash",None)==request_hash
    if controller is not None:
        _cache(controller).put(("result",request_hash),encoded,len(encoded.encode("utf-8")))
    return {"status":"RECORDED","request_hash":request_hash,"analysis_identity":request["identity"],"result_hash":result_hash,
            "original_state_version":identity.state_version,"historical":not current,"eligible_for_current_display":bool(selected),
            "automatic_action":False,"archive_path":str(store.path),"record":record}


def record_analysis(controller,identity,result,request_hash=None):
    return _record(controller.store,identity,result,request_hash,controller)


def record_archived_analysis(archive_path,identity,result,request_hash=None):
    from qqq_sim.config import writable_path
    from qqq_sim.storage.events import EventStore
    target=writable_path(archive_path)
    if not target.is_file():
        raise BrokerError("ANALYSIS_ARCHIVE_NOT_FOUND",path=str(target))
    with EventStore(target) as store:
        return _record(store,identity,result,request_hash)


def archive_unavailable(controller,draft,selection_hash,result,*,current_position=False,request_context=None):
    c=controller
    campaign=c.broker.ledger.campaigns.get(getattr(draft,"campaign_id",None))
    identity=AnalysisIdentity(c.session_id,c.branch_id,c.version,c.snapshot.snapshot_id,selection_hash,digest(primitive(campaign)),
                              digest({"draft":primitive(draft),"current_position":current_position,"unavailable":primitive(result)}),c._pack_hash,digest(c.config.rule_profile))
    prepared=register_analysis_request(c,{"identity":identity,"unavailable_inputs":True},draft=draft,current_position=current_position,request_context=request_context)
    record_analysis(c,identity,result,prepared["request_hash"])
    return {**result,"analysis_identity":primitive(identity),"request_hash":prepared["request_hash"],"payload_hash":prepared["payload_hash"],
            "archive_path":prepared["archive_path"],"selection_hash":selection_hash,"state_version":c.version,"already_archived":True}
