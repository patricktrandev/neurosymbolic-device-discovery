"""Root conftest: add src/ to sys.path so pytest can resolve all packages."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
