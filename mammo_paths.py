from pathlib import Path
import os
import platform

# Base directory where app is running
_BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = _BASE_DIR / "data"
COORDINATES = DATA_DIR / "coordinates"
LABEL_CSV = DATA_DIR / "labels.csv"
MASKS = DATA_DIR / "masks"
POLYGONS = DATA_DIR / "polygons"

def ensure_data_dirs():
    """Create all necessary data directories"""
    COORDINATES.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(parents=True, exist_ok=True)
    POLYGONS.mkdir(parents=True, exist_ok=True)

# Create folders
ensure_data_dirs()

# User file path - platform specific
if platform.system() == "Darwin":  # macOS
    APP_SUPPORT = Path.home() / "Library" / "Application Support" / "MammoX"
    APP_SUPPORT.mkdir(parents=True, exist_ok=True)
    USER_FILE = APP_SUPPORT / "users.json"
else:
    USER_FILE = _BASE_DIR / "users.json"