"""Run the complete implemented CIC-IDS2017 pipeline."""

from run_all_steps import main
from pathlib import Path
import os


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
