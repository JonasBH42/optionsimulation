from .parameters import ParameterRegistry
from .models import DataError


class ProductRegistry(ParameterRegistry):
    def resolve(self, contract_identity, effective_date):
        identity = getattr(contract_identity, "contract_id", str(contract_identity))
        record = self.get("multiplier/deliverable", "QQQ", effective_date, contract_id=identity)
        if record is None:
            raise DataError("CONTRACT_METADATA_MISSING", identity)
        return record
