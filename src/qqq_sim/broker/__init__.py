"""Causal, local-only broker simulation with explicit scenario profiles."""
from .models import BrokerError, Contract, Quote, OrderLeg, OrderDraft, Fill
from .fees import FeeEngine, FeeProfile


def __getattr__(name):
    if name == 'Broker':
        from .service import Broker
        return Broker
    raise AttributeError(name)

