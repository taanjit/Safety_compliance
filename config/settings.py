"""
Central project settings for the PPE Detection project.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATASET_YAML = CONFIG_DIR / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"

# Scheduler directories
INPUT_DIR = PROJECT_ROOT / "files" / "input"
OUTPUT_DIR = PROJECT_ROOT / "files" / "output"
LOG_FILE = PROJECT_ROOT / "logs" / "processed.log"

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ──────────────────────────────────────────────
# Model Defaults
# ──────────────────────────────────────────────
DEFAULT_MODEL = "yolo26n.pt"       # Pretrained YOLO model checkpoint
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_CONF = 0.25                # Confidence threshold for inference
DEFAULT_IOU = 0.45                 # IoU threshold for NMS
DEFAULT_DEVICE = ""                # "" = auto-detect (CUDA if available, else CPU)
DEFAULT_FRAME_SKIP = 10            # Process every Nth frame for video inference
DEFAULT_PROJECT = str(RUNS_DIR / "detect")
DEFAULT_NAME = "ppe_detection"

# ──────────────────────────────────────────────
# Class Definitions
# ──────────────────────────────────────────────
CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}

# Classes that represent WORN PPE (compliant)
WORN_PPE_CLASSES = {0, 1, 2, 3, 4}  # helmet, gloves, vest, boots, goggles

# Classes that represent MISSING PPE (non-compliant / violation)
MISSING_PPE_CLASSES = {7, 8, 9, 10}  # no_helmet, no_goggle, no_gloves, no_boots

# Neutral classes
NEUTRAL_CLASSES = {5, 6}  # none, Person

# ──────────────────────────────────────────────
# Visualization Colors  (BGR format for OpenCV)
# ──────────────────────────────────────────────
COLOR_COMPLIANT = (0, 200, 0)       # Green  — worn PPE
COLOR_VIOLATION = (0, 0, 220)       # Red    — missing PPE
COLOR_NEUTRAL = (200, 180, 0)       # Cyan   — Person / none
