import sys
from pathlib import Path

# Make the project root importable, so tests can use `from app.main import app`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))