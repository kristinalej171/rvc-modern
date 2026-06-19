"""
Точка входа для запуска через `python -m rvc`.

Использование:
    python -m rvc --port 7860
    python -m rvc --host 0.0.0.0 --port 7860 --share
"""

import sys

from rvc.main import main

if __name__ == "__main__":
    sys.exit(main())
