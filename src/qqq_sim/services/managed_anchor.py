"""Select the managed market law without changing the unchanged-HOLD law."""
from datetime import datetime
from qqq_sim.risk.contracts import RiskError,ReferenceJob,RiskLeg,canonical,utc_us


def managed_job_from_known_anchor(hold_job,universe,hard_cutoff_utc_us,snapshot,parameters,calendar,pack=None):
    from .adapters import broker_contract,parameters_for_expiry
    base=ReferenceJob.from_dict(hold_job)
    legs=tuple(leg if isinstance(leg,RiskLeg) else RiskLeg.from_dict(leg) for leg in universe)
    audit={'status':'UNSUPPORTED','reason':'MANAGED_ANCHOR_UNAVAILABLE',
           'hold_fixing_utc_us':base.fixing_utc_us,'hard_cutoff_utc_us':hard_cutoff_utc_us,
           'hold_sigma_path':base.sigma_path,'hold_job_hash':base.hash,
           'selection_rule':'KNOWN_ADMITTED_EXPIRY_NEAREST_HARD_CUTOFF_TIES_LATER'}
    try:
        if not legs:raise RiskError('EMPTY_KNOWN_UNIVERSE')
        anchor=min({leg.fixing_utc_us for leg in legs},key=lambda when:(abs(when-hard_cutoff_utc_us),-when))
        audit.update(anchor_fixing_utc_us=anchor,maturity_mismatch_us=anchor-hard_cutoff_utc_us)
        keys={leg.contract_id for leg in legs if leg.fixing_utc_us==anchor}
        entries=[entry for entry in snapshot.quotes if broker_contract(entry.contract,calendar).key in keys]
        if not entries:raise RiskError('ANCHOR_QUOTES_NOT_IN_SNAPSHOT')
        expiry=entries[0].contract.expiration
        scoped=parameters_for_expiry(parameters,expiry)
        known=datetime.fromisoformat(scoped['known_at'])
        if known.tzinfo is None or known>snapshot.asof:raise RiskError('FUTURE_PARAMETER')
        if utc_us(datetime.fromisoformat(scoped['fixing_at']))!=anchor:
            raise RiskError('INCOMPATIBLE_ANCHOR_FIXING')
        sigma=scoped.get('sigma_path')
        if sigma is None:
            from .path_iv import local_atm_path_iv
            weight=sum(abs(leg.quantity)*leg.multiplier for leg in base.signed_legs)
            eps=min(.005,1e-4*max(1.,abs(float(base.target_usd))))/max(weight,1)
            sigma,evidence=local_atm_path_iv(snapshot,expiry,scoped,pack,eps)
        else:
            sigma=float(sigma)
            evidence={'source':'EXPLICIT_ASOF_EXPIRY_PATH_IV','sigma_path':sigma,
                      'known_at_utc_us':utc_us(known),'expiry':expiry.isoformat(),
                      'parameter_source':scoped.get('source','EXPLICIT_SCENARIO')}
        result=dict(hold_job)
        result.update(sigma_path=sigma,rate_cc=scoped['rate_cc'],dividend_yield_cc=scoped['dividend_yield_cc'])
        audit.update(status='OK',reason=None,anchor_expiry=expiry.isoformat(),sigma_managed=sigma,
                     rate_cc=scoped['rate_cc'],dividend_yield_cc=scoped['dividend_yield_cc'],path_iv_evidence=evidence)
        result['parameter_provenance']=[*hold_job.get('parameter_provenance',()),
            {'source':'P50_M04_MANAGED_ANCHOR','known_at_utc_us':utc_us(known),'anchor':dict(audit)}]
        managed=ReferenceJob.from_dict(result)
        audit['managed_job_hash']=managed.hash
        return canonical(managed),audit
    except (RiskError,ValueError,TypeError,KeyError) as error:
        audit['cause']=getattr(error,'code','MANAGED_ANCHOR_INPUTS_INVALID')
        audit['message']=str(error)
        return None,audit
