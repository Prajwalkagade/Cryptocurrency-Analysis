import os
from pathlib import Path

DATA_DIR = Path("data")

def ensure_data_folders():
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
