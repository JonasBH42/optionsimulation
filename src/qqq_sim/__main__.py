from __future__ import annotations
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    from .config import configure_environment
    configure_environment()
    from .cli import main
    raise SystemExit(main())
