"""Makes the `aitext` package importable for tests without requiring an editable
install (`pip install -e .`), so `pytest` works straight after cloning the repo."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
