"""Run decision engine tests directly with proper path setup."""

import sys
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Import pytest after path is set
import pytest

sys.exit(pytest.main(["tests/decision_engine/", "-v", "--tb=short"]))
