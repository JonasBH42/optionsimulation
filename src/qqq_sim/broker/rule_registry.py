"""Local dated rule records; missing historic components never inherit today's rules."""
from dataclasses import dataclass
from datetime import date,datetime
from pathlib import Path
import json
from .models import BrokerError,fingerprint,canonical
from .fees import FeeProfile

REQUIRED_COMPONENTS=frozenset({'option_commission','clearing','orf','taf','section31','exercise_assignment',
                               'collateral','settlement','account_permissions','day_trading'})


@dataclass(frozen=True)
class RuleRecord:
    profile_id: str
    effective_start: date
    effective_end: date
    known_at: datetime
    components: dict
    mode: str
    provenance: str

    @property
    def hash(self): return fingerprint(self)


class RuleRegistry:
    def __init__(self,records=()): self.records=list(records)

    def resolve(self,economic_date,asof,mode='fixed_rules_counterfactual'):
        if mode=='fixed_rules_counterfactual':
            p=FeeProfile()
            return {'id':p.id,'fee_profile':p,'hash':p.hash,'mode':mode,
                    'rule_asof':p.rule_asof,'qualification':'Published component economics with explicit simulation conventions',
                    'account':'OPTIONS_COLLATERAL_QQQ_V1','pdt_counter':'NOT_APPLICABLE_PROFILE'}
        if mode!='historical_rules': raise BrokerError('UNKNOWN_RULE_MODE')
        eligible=[r for r in self.records if r.mode==mode and r.effective_start<=economic_date<=r.effective_end and r.known_at<=asof]
        if not eligible: raise BrokerError('BROKER_RULE_HISTORY_MISSING')
        selected=max(eligible,key=lambda r:(r.known_at,r.profile_id))
        missing=sorted(REQUIRED_COMPONENTS-set(selected.components))
        if missing: raise BrokerError('BROKER_RULE_HISTORY_MISSING',missing_components=missing)
        return selected

    @classmethod
    def from_json(cls,path):
        # The external file is read-only runtime data.
        records=[]
        for raw in json.loads(Path(path).read_text(encoding='utf-8')):
            raw=dict(raw)
            raw['effective_start']=date.fromisoformat(raw['effective_start'])
            raw['effective_end']=date.fromisoformat(raw['effective_end'])
            raw['known_at']=datetime.fromisoformat(raw['known_at'])
            records.append(RuleRecord(**raw))
        return cls(records)

