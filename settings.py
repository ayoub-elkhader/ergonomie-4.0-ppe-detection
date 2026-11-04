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
# Classes of the soudage_ia dataset that mean a safety rule is NOT respected.
NON_COMPLIANT_CLASSES = {
    "sans_casque": "Welding helmet missing",
    "sans_gants": "Gloves missing",
    "sans_tablier": "Apron missing",
    "tablier_ouvert": "Apron open",
    "chaussure_inadequat": "Unsuitable shoes",
    "presence_accessoires": "Accessories worn (jewellery, watch...)",
    "presence_cheveux": "Hair not covered",
}
