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

# COCO-pretrained YOLOv8n, used ONLY to count the people in the frame
# (the PPE model has no "person" class). Downloaded by Ultralytics if missing.
PERSON_MODEL = MODEL_DIR / "yolov8n.pt"
PERSON_CLASS_ID = 0           # "person" in COCO
PERSON_CONFIDENCE = 0.40

# -------------------------------------------------------------------- PPE
# Detector scores are NOT probabilities of compliance. They are only used as
# thresholds to decide which detections count as evidence:
#   score >= decision threshold (slider in the app)  -> confirmed evidence
#   UNCERTAIN_CONFIDENCE <= score < decision threshold -> uncertain evidence
#   score < UNCERTAIN_CONFIDENCE                     -> ignored
DEFAULT_DECISION_CONFIDENCE = 0.50
UNCERTAIN_CONFIDENCE = 0.25

# Scope: the decision is made for ONE operator per image. Detections are
# not associated with individual people, so frames with 0 or several people
# are reported as "evaluation incomplete".
#
# Required equipment: (label, classes proving it is worn, classes proving it
# is missing or badly worn). Absence of a "missing" detection is never taken
# as proof that the item is worn.
PPE_CHECKLIST = [
    ("Welding helmet", ["avec_casque"], ["sans_casque"]),
    ("Gloves", ["avec_gants"], ["sans_gants"]),
    ("Apron", ["tablier_ferme"], ["sans_tablier", "tablier_ouvert"]),
    ("Safety shoes", ["chaussure_adequat"], ["chaussure_inadequat"]),
]

# Extra hazards: a confirmed detection is an anomaly. Their absence cannot be
# verified (hair or jewellery may simply be hidden).
HAZARD_CLASSES = {
    "presence_accessoires": "accessories worn (jewellery, watch...)",
    "presence_cheveux": "hair not covered",
}

# Human-readable messages for "missing / badly worn" classes
VIOLATION_LABELS = {
    "sans_casque": "no helmet",
    "sans_gants": "no gloves",
    "sans_tablier": "no apron",
    "tablier_ouvert": "apron open",
    "chaussure_inadequat": "unsuitable shoes",
}

# Classes that sit on the head: two or more confirmed ones suggest several people.
HEAD_CLASSES = ("avec_casque", "sans_casque")

# ---------------------------------------------------------------- Posture
# Minimum MediaPipe landmark visibility for a point to be used.
MIN_LANDMARK_VISIBILITY = 0.5
# Left/right difference (degrees) above which an asymmetry is reported.
ASYMMETRY_DEG = 15

# EXPLORATORY thresholds (degrees): (low/elevated limit, elevated/high limit).
# The trunk and neck bands reuse the angle limits of RULA (McAtamney & Corlett,
# Applied Ergonomics 24(2), 1993) for orientation only: the quantities measured
# here are approximations from a single camera, and no RULA/REBA score is
# computed. Adjust them to your own ergonomic reference.
POSTURE_THRESHOLDS = {
    "trunk_inclination": (20, 60),
    "neck_flexion": (10, 20),
    "hip_flexion": (45, 90),
    "knee_flexion": (30, 60),
}
# Indicators used for the overall level; the others are shown for information.
POSTURE_RATED = ("trunk_inclination", "neck_flexion")
