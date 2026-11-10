import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qqq_sim.config import configure_environment
configure_environment()
