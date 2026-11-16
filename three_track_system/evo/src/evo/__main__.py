"""支持 `python -m evo` 直接启动。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
