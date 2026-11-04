"""Central configuration: paths, sources and PPE classes."""
from pathlib import Path

# Absolute project root, so the app works whatever the current directory is.
ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- Sources
IMAGE = "Image"
VIDEO = "Video"
WEBCAM = "Webcam"
RTSP = "RTSP"
SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP]

# ----------------------------------------------------------------- Images
IMAGES_DIR = ROOT / "images"
DEFAULT_IMAGE = IMAGES_DIR / "epi1.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "epi1cor.jpg"

# ----------------------------------------------------------------- Videos
# Every .mp4 / .avi / .mov file dropped in videos/ shows up in the app.
VIDEO_DIR = ROOT / "videos"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def list_videos():
    """Return {display name: path} for the videos found in VIDEO_DIR."""
    if not VIDEO_DIR.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(VIDEO_DIR.iterdir())
        if p.suffix.lower() in VIDEO_EXTENSIONS
    }


# ------------------------------------------------------------------ Model
MODEL_DIR = ROOT / "weights"
DETECTION_MODEL = MODEL_DIR / "best.pt"

# Webcam index used by OpenCV (0 = default camera)
WEBCAM_PATH = 0

# -------------------------------------------------------------------- PPE
# Required equipment: (label, classes proving it is worn, classes proving a violation).
# An item with neither kind of detection is reported as "not visible".
PPE_CHECKLIST = [
    ("Welding helmet", ["avec_casque"], ["sans_casque"]),
    ("Gloves", ["avec_gants"], ["sans_gants"]),
    ("Apron", ["tablier_ferme"], ["sans_tablier", "tablier_ouvert"]),
    ("Safety shoes", ["chaussure_adequat"], ["chaussure_inadequat"]),
]

# Extra hazards: detected = violation, not detected = fine.
HAZARD_CLASSES = {
    "presence_accessoires": "Accessories worn (jewellery, watch...)",
    "presence_cheveux": "Hair not covered",
}

# Human-readable messages for violation classes
VIOLATION_LABELS = {
    "sans_casque": "no helmet",
    "sans_gants": "no gloves",
    "sans_tablier": "no apron",
    "tablier_ouvert": "apron open",
    "chaussure_inadequat": "unsuitable shoes",
}
