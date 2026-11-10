"""Reference kernel identity in source installations and compiled distributions."""
import hashlib
from pathlib import Path
from ..contracts import RiskError,stable_hash


FILES=("contracts.py","payoff.py","pricing_fast.py","offline/hit_oracle.py","offline/pricing_oracle.py",
       "offline/shards.py","offline/region_store.py","offline/european_regions.py")


def implementation_identity():
    root=Path(__file__).parents[1]
    from qqq_sim._build_info import SOURCE_HASHES
    hashes={}
    for relative in FILES:
        path=root/relative
        key="src/qqq_sim/risk/"+relative
        if path.is_file():
            hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        elif key in SOURCE_HASHES:
            hashes[key]=SOURCE_HASHES[key]
        else:
            raise RiskError("IMPLEMENTATION_IDENTITY_UNAVAILABLE",source=key)
    return {"sha256":stable_hash(hashes),"files":hashes}
