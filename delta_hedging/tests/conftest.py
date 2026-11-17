import sys
from pathlib import Path

# Make `import hedging` work when pytest is run from the project folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
