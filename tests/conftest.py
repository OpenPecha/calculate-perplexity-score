import sys
from pathlib import Path

# Add the source directory to sys.path so the scripts can be imported directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "calculate-perplexity-score"))
