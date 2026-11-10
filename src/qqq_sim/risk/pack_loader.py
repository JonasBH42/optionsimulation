"""Non-executable immutable model-pack loader with artifact byte verification."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import numpy as np
from .contracts import RiskError, stable_hash
from .pricing_fast import PriceBasis,PRICE_TREE_SCHEMA


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


class ModelPack:
    def __init__(self, path, manifest, arrays, validation):
        self.path = Path(path)
        self.manifest = MappingProxyType(manifest)
        self.arrays = MappingProxyType(arrays)
        self.validation = MappingProxyType(validation)
        self.pack_id = manifest["pack_id"]
        self.manifest_hash = stable_hash(manifest)
        self.price_basis = None
        tree_names=('price_tree_nodes','price_tree_bounds')
        has_tree=any(name in arrays for name in tree_names)
        if has_tree and (not all(name in arrays for name in tree_names) or manifest.get('price_tree_schema')!=PRICE_TREE_SCHEMA):
            raise RiskError('PRICE_TREE_INVALID',detail='Incomplete tree artifacts or incompatible schema')
        if all(k in arrays for k in ("price_bounds", "price_coefficients", "price_errors")):
            for name in ('price_bounds','price_coefficients','price_errors','price_greek_errors'):
                if name in arrays and arrays[name].dtype!=np.dtype('<f8'):
                    raise RiskError('INVALID_PRICE_BASIS',artifact=name,detail='Stored price arrays must be little-endian float64')
            self.price_basis = PriceBasis(arrays["price_bounds"], arrays["price_coefficients"], arrays["price_errors"], arrays.get("price_greek_errors"),
                                         tree_nodes=arrays.get('price_tree_nodes'),tree_bounds=arrays.get('price_tree_bounds'))
        elif has_tree:
            raise RiskError('PRICE_TREE_INVALID',detail='Tree has no complete accepted-cell basis')
        if validation.get('capabilities',{}).get('price',{}).get('qualified') and (self.price_basis is None or self.price_basis.tree is None):
            raise RiskError('PRICE_DECISION_TREE_MISSING')

    @classmethod
    def load(cls, path, required_profiles=None):
        path = Path(path)
        if not (path/"manifest.json").is_file():
            raise RiskError("PACK_MISSING", path=str(path))
        try:
            manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RiskError("PACK_MANIFEST_INVALID") from exc
        if manifest.get("schema_version") != "QQQ_P50_PACK_V1" or not manifest.get("pack_id"):
            raise RiskError("PACK_SCHEMA_UNSUPPORTED")
        receipt = path/"manifest.sha256"
        if not receipt.is_file() or receipt.read_text().strip() != file_hash(path/"manifest.json"):
            raise RiskError("PACK_INTEGRITY_FAILURE", artifact="manifest.json")
        arrays = {}
        for name, artifact in manifest.get("artifacts", {}).items():
            relative = Path(artifact["file"])
            target = (path/relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(path.resolve()) or not target.is_file():
                raise RiskError("PACK_INVALID_ARTIFACT_PATH", artifact=name)
            if target.stat().st_size != artifact["bytes"] or file_hash(target) != artifact["sha256"]:
                raise RiskError("PACK_INTEGRITY_FAILURE", artifact=name)
            if target.suffix == ".npy":
                try:
                    array = np.load(target, mmap_mode="r", allow_pickle=False)
                except ValueError as exc:
                    raise RiskError("PACK_UNSAFE_ARRAY", artifact=name) from exc
                if array.dtype.hasobject or array.dtype.kind not in "fibu" or not np.all(np.isfinite(array)):
                    raise RiskError("PACK_INVALID_ARRAY", artifact=name)
                arrays[name] = array
        validation_path = path/"validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {"qualified": False, "reason": "VALIDATION_MISSING"}
        if not isinstance(validation,dict) or not isinstance(validation.get("capabilities",{}),dict):
            raise RiskError("PACK_VALIDATION_SCHEMA_INVALID")
        validation_receipt = path/"validation.sha256"
        if not validation_receipt.is_file() or validation_receipt.read_text().strip() != file_hash(validation_path):
            raise RiskError("PACK_INTEGRITY_FAILURE", artifact="validation.json")
        if validation.get("manifest_hash") != stable_hash(manifest):
            raise RiskError("PACK_VALIDATION_MISMATCH")
        for key, value in (required_profiles or {}).items():
            if manifest.get("profiles", {}).get(key) != value:
                raise RiskError("PACK_PROFILE_MISMATCH", field=key)
        return cls(path, manifest, arrays, validation)

    def require_qualified(self, capability="hold"):
        gate = self.validation.get("capabilities", {}).get(capability, {})
        if not gate.get("qualified", False):
            raise RiskError("PACK_NOT_QUALIFIED", capability=capability, reason=gate.get("reason", "CALIBRATION_REQUIRED"))
        if capability=="price":
            if self.price_basis is None:
                raise RiskError("PRICE_BASIS_MISSING")
            if self.price_basis.tree is None:
                raise RiskError('PRICE_DECISION_TREE_MISSING')
            return
        prefix = "hold" if capability == "hold" else "q_eval"
        from .feature_layout import FEATURE_SCHEMA,HOLD_FEATURE_COUNT,MANAGED_FEATURE_COUNT,feature_contract
        if self.manifest.get("feature_schema")!=FEATURE_SCHEMA:
            raise RiskError("FEATURE_SCHEMA_MISMATCH")
        if stable_hash(self.manifest.get('feature_contract'))!=stable_hash(feature_contract()):
            raise RiskError('FEATURE_CONTRACT_MISMATCH')
        if capability=='hold':
            if (self.manifest.get('hold_label_schema',{}).get('version')!='QQQ_HOLD_LABEL_V2'
                or self.manifest.get('hold_loss_schema',{}).get('version')!='QQQ_HOLD_LOSS_V2'):
                raise RiskError('HOLD_TRAINING_CONTRACT_MISMATCH')
            if not gate.get('within_bin_validation',{}).get('qualified',False):
                raise RiskError('HOLD_WITHIN_BIN_VALIDATION_MISSING')
        expected_inputs=HOLD_FEATURE_COUNT if capability=="hold" else MANAGED_FEATURE_COUNT
        expected_outputs = 579 if capability == "hold" else 1
        layers = [(f"{prefix}_w{i}", f"{prefix}_b{i}") for i in range(4)]
        if any(w not in self.arrays or b not in self.arrays for w, b in layers):
            raise RiskError("PACK_NETWORK_MISSING")
        previous = expected_inputs
        for i, (w, b) in enumerate(layers):
            weight, bias = self.arrays[w], self.arrays[b]
            output = (256, 256, 128, expected_outputs)[i]
            if weight.dtype != np.float64 or bias.dtype != np.float64 or weight.ndim != 2 or bias.shape != (output,) or weight.shape[1] != output or previous is not None and weight.shape[0] != previous:
                raise RiskError("PACK_NETWORK_SCHEMA_MISMATCH")
            previous = output
        for name in (prefix+"_mean",prefix+"_std"):
            if name not in self.arrays or self.arrays[name].dtype!=np.float64 or self.arrays[name].shape!=(expected_inputs,):
                raise RiskError("PACK_NETWORK_SCHEMA_MISMATCH",artifact=name)
        if np.any(self.arrays[prefix+"_std"]<1e-8):raise RiskError("PACK_NETWORK_SCHEMA_MISMATCH")
        if capability=="managed":
            previous=expected_inputs
            for index,output in enumerate((256,256,128,1)):
                w,b=f"q_score_w{index}",f"q_score_b{index}"
                if w not in self.arrays or b not in self.arrays or self.arrays[w].shape!=(previous,output) or self.arrays[b].shape!=(output,) or self.arrays[w].dtype!=np.float64 or self.arrays[b].dtype!=np.float64:
                    raise RiskError("PACK_NETWORK_SCHEMA_MISMATCH",artifact=w)
                previous=output
            for name,shape in (("universe_w0",(16,64)),("universe_b0",(64,)),("universe_w1",(64,64)),("universe_b1",(64,))):
                for kind in ("","q_score_","q_eval_"):
                    key=kind+name
                    if key not in self.arrays or self.arrays[key].shape!=shape or self.arrays[key].dtype!=np.float64:
                        raise RiskError("PACK_NETWORK_SCHEMA_MISMATCH",artifact=key)
            for name in ("q_score_mean","q_score_std","managed_mean","managed_std"):
                if name not in self.arrays or self.arrays[name].shape!=(MANAGED_FEATURE_COUNT,) or self.arrays[name].dtype!=np.float64:
                    raise RiskError("PACK_NETWORK_SCHEMA_MISMATCH",artifact=name)
        if gate.get("independent_cases", 0) < 200 or gate.get("pass_fraction", 0) < .99:
            raise RiskError("PACK_CALIBRATION_INSUFFICIENT")
        evidence=capability+"_validation_features"
        if evidence not in self.arrays or len(self.arrays[evidence])<200 or self.arrays[evidence].ndim!=2 or self.arrays[evidence].shape[1]!=expected_inputs:
            raise RiskError("PACK_VALIDATION_CASE_ARRAYS_MISSING")


def load_pack(path):
    return ModelPack.load(path)
