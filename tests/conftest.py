import sys
from pathlib import Path

# tests import spec2desc, train/ and eval/ as plain scripts
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "train"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
