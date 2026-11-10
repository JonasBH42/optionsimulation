"""Importable, guarded standalone Windows launcher."""
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    from qqq_sim.config import configure_environment
    configure_environment()
    from qqq_sim.cli import main
    raise SystemExit(main())
