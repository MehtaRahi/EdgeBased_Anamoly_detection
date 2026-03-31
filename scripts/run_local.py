import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.run_all_local import main

if __name__ == "__main__":
    main()