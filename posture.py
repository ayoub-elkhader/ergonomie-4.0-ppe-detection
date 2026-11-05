"""Exploratory posture indicators from MediaPipe Pose (single camera).

Measured quantities (computed separately for the left and right side):

trunk_inclination  Angle between the hip->shoulder segment and the UPWARD
                   VERTICAL OF THE IMAGE, in the image plane (2D, pixels).
                   It equals the trunk inclination relative to gravity only if
                   (a) the camera is level (no roll / pitch) and (b) the person
                   is seen from the side. A front view hides forward bending.
                   No calibration is applied; tilt the camera and the value is
                   biased by the tilt angle. Unsigned: leaning back counts too.
neck_flexion       Angle between the shoulder->ear segment and the
                   hip->shoulder segment (3D world landmarks). 0 when ear,
                   shoulder and hip are aligned. It does not distinguish
                   flexion from extension, and the ear is used as a proxy for
                   the head axis.
hip_flexion        180 deg minus the angle shoulder-hip-knee (3D): angle
                   between thigh and torso. Standing = 0, seated upright ~ 90.
                   Information only (sitting is not a risk in itself).
knee_flexion       180 deg minus the angle hip-knee-ankle (3D).
                   Information only.

3D angles are between body segments, so they do not depend on the camera
orientation, but MediaPipe's depth estimate from a single image is
approximate. A side needs every point it uses with visibility >=
settings.MIN_LANDMARK_VISIBILITY; otherwise that side is "unavailable".
When both sides are available, the larger value is used and a left/right
difference above settings.ASYMMETRY_DEG is reported.

These are exploratory indicators with configurable thresholds
(settings.POSTURE_THRESHOLDS). They are NOT a RULA/REBA score and NOT a
validated ergonomic assessment.
"""
import time

import cv2
import numpy as np

import settings

# MediaPipe Pose landmark indices: (left, right)
EAR, SHOULDER, HIP, KNEE, ANKLE = (7, 8), (11, 12), (23, 24), (25, 26), (27, 28)

INDICATORS = {
    "trunk_inclination": "Trunk inclination (vs image vertical)",
    "neck_flexion": "Neck flexion (vs trunk)",
    "hip_flexion": "Hip flexion (thigh-torso)",
    "knee_flexion": "Knee flexion",
}
SHORT_LABELS = {"trunk_inclination": "Trunk incl.", "neck_flexion": "Neck flex.",
                "hip_flexion": "Hip flex.", "knee_flexion": "Knee flex."}
# points needed by each indicator (one index pair per point, used per side)
REQUIRED_POINTS = {
    "trunk_inclination": (SHOULDER, HIP),
    "neck_flexion": (EAR, SHOULDER, HIP),
    "hip_flexion": (SHOULDER, HIP, KNEE),
    "knee_flexion": (HIP, KNEE, ANKLE),
}

LOW, ELEVATED, HIGH, UNAVAILABLE = 0, 1, 2, -1
LEVEL_TEXT = {LOW: "low", ELEVATED: "elevated", HIGH: "high", UNAVAILABLE: "unavailable"}
LEVEL_BGR = {LOW: (80, 200, 60), ELEVATED: (0, 170, 255), HIGH: (40, 40, 235),
             UNAVAILABLE: (170, 170, 170)}
OVERALL_TEXT = {
    LOW: "all rated indicators below thresholds",
    ELEVATED: "indicator above the first threshold",
    HIGH: "indicator above the second threshold",
    UNAVAILABLE: "rated indicators unavailable",
}
EPS = 1e-6


# ----------------------------------------------------------------- geometry
def angle_between(u, v):
    """Unsigned angle in degrees between two vectors; None if one is ~zero."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < EPS or nv < EPS or not np.isfinite(nu * nv):
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


def trunk_inclination_2d(shoulder_px, hip_px):
    """Angle between hip->shoulder and the image 'up' direction (y axis points down)."""
    return angle_between(np.asarray(shoulder_px) - np.asarray(hip_px), (0.0, -1.0))


def neck_flexion_3d(ear, shoulder, hip):
    return angle_between(np.asarray(ear) - shoulder, np.asarray(shoulder) - hip)


def joint_flexion_3d(a, joint, b):
    """180 - angle a-joint-b: 0 when the three points are aligned."""
    ang = angle_between(np.asarray(a) - joint, np.asarray(b) - joint)
    return None if ang is None else 180.0 - ang


def _side_value(name, side, pts2d, pts3d, vis, min_vis):
    idx = [pair[side] for pair in REQUIRED_POINTS[name]]
    if any(not np.isfinite(vis[i]) or vis[i] < min_vis for i in idx):
        return None
    if name == "trunk_inclination":
        return trunk_inclination_2d(pts2d[idx[0]], pts2d[idx[1]])
    if name == "neck_flexion":
        return neck_flexion_3d(pts3d[idx[0]], pts3d[idx[1]], pts3d[idx[2]])
    return joint_flexion_3d(pts3d[idx[0]], pts3d[idx[1]], pts3d[idx[2]])


def rate(name, value, thresholds=None):
    if value is None:
        return UNAVAILABLE
    low, high = (thresholds or settings.POSTURE_THRESHOLDS)[name]
    if value <= low:
        return LOW
    return ELEVATED if value <= high else HIGH


def compute_indicators(pts2d, pts3d, vis, min_vis=None, asym_deg=None, thresholds=None):
    """Indicators from landmark arrays.

    pts2d: (33, 2) image coordinates in PIXELS (y pointing down)
    pts3d: (33, 3) world coordinates (metres, any orientation)
    vis:   (33,) visibility in [0, 1]
    Returns {name: {"left", "right", "value", "asymmetry", "level"}} and the
    overall level of the rated indicators.
    """
    min_vis = settings.MIN_LANDMARK_VISIBILITY if min_vis is None else min_vis
    asym_deg = settings.ASYMMETRY_DEG if asym_deg is None else asym_deg
    pts2d, pts3d, vis = np.asarray(pts2d, float), np.asarray(pts3d, float), np.asarray(vis, float)

    out = {}
    for name in INDICATORS:
        left = _side_value(name, 0, pts2d, pts3d, vis, min_vis)
        right = _side_value(name, 1, pts2d, pts3d, vis, min_vis)
        sides = [v for v in (left, right) if v is not None]
        value = max(sides) if sides else None
        asym = left is not None and right is not None and abs(left - right) > asym_deg
        out[name] = {"left": left, "right": right, "value": value,
                     "asymmetry": asym, "level": rate(name, value, thresholds)}

    rated = [out[n]["level"] for n in settings.POSTURE_RATED]
    overall = UNAVAILABLE if all(l == UNAVAILABLE for l in rated) else max(rated)
    return {"indicators": out, "overall": overall}


# --------------------------------------------------------------- MediaPipe
def landmarks_to_arrays(pose_landmarks, world_landmarks, width, height):
    """Convert MediaPipe results to (pts2d px, pts3d, visibility) arrays."""
    lm2 = pose_landmarks.landmark
    lm3 = world_landmarks.landmark
    pts2d = np.array([[p.x * width, p.y * height] for p in lm2])
    pts3d = np.array([[p.x, p.y, p.z] for p in lm3])
    vis = np.array([getattr(p, "visibility", 1.0) for p in lm2])
    return pts2d, pts3d, vis


def create_pose(static_image=False):
    """MediaPipe Pose estimator (video stream by default, or single images)."""
    import mediapipe as mp
    return mp.solutions.pose.Pose(
        static_image_mode=static_image,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


class PostureEstimator:
    """Runs MediaPipe on the ORIGINAL frame and keeps the last landmarks."""

    def __init__(self, static_image=False):
        self.pose = create_pose(static_image)
        self.landmarks = None

    def estimate(self, frame_bgr):
        """Return the assessment dict (None if nobody is detected)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.pose.process(rgb)
        self.landmarks = res.pose_landmarks
        if res.pose_landmarks is None or res.pose_world_landmarks is None:
            return None
        h, w = frame_bgr.shape[:2]
        return compute_indicators(*landmarks_to_arrays(res.pose_landmarks,
                                                       res.pose_world_landmarks, w, h))

    def draw(self, image_bgr, assessment):
        """Draw skeleton (from the last estimate) and the indicator panel."""
        import mediapipe as mp
        draw, pose = mp.solutions.drawing_utils, mp.solutions.pose
        h, w = image_bgr.shape[:2]
        scale = max(0.8, min(w, h) / 500)
        if self.landmarks is not None:
            t = max(2, int(3 * scale))
            draw.draw_landmarks(image_bgr, self.landmarks, pose.POSE_CONNECTIONS,
                                draw.DrawingSpec((255, 255, 255), t, t),
                                draw.DrawingSpec((255, 200, 0), t))
        _panel(image_bgr, panel_lines(assessment), scale)
        return image_bgr

    def close(self):
        self.pose.close()


def format_value(ind):
    if ind["value"] is None:
        return "unavailable"
    if ind["asymmetry"]:
        return f"L {ind['left']:.0f} / R {ind['right']:.0f} deg"
    return f"{ind['value']:.0f} deg"


def panel_lines(assessment):
    """Short lines for the image overlay (details are in the app's table)."""
    if assessment is None:
        return [("Posture: no person", LEVEL_BGR[UNAVAILABLE])]
    lines = [(f"Posture: {LEVEL_TEXT[assessment['overall']]}", LEVEL_BGR[assessment["overall"]])]
    for name in INDICATORS:
        ind = assessment["indicators"][name]
        info = "" if name in settings.POSTURE_RATED else " (i)"
        value = "n/a" if ind["value"] is None else f"{ind['value']:.0f} deg"
        asym = " L/R!" if ind["asymmetry"] else ""
        lines.append((f"{SHORT_LABELS[name]}: {value}{asym}{info}", LEVEL_BGR[ind["level"]]))
    return lines


def _panel(image, lines, scale):
    """Semi-transparent dark panel with coloured text lines (top-left)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.55 * scale, max(1, int(round(1.5 * scale)))
    pad, gap = int(10 * scale), int(10 * scale)
    sizes = [cv2.getTextSize(text, font, fs, th)[0] for text, _ in lines]
    width = min(image.shape[1], max(w for w, _ in sizes) + 2 * pad)
    line_h = max(h for _, h in sizes) + gap
    height = min(image.shape[0], line_h * len(lines) + pad)
    roi = image[:height, :width]
    roi[:] = (roi * 0.25 + np.array((25, 25, 25)) * 0.75).astype(np.uint8)
    for i, (text, color) in enumerate(lines):
        y = pad + (i + 1) * line_h - gap // 2
        cv2.putText(image, text, (pad, y), font, fs, color, th, cv2.LINE_AA)


# ---------------------------------------------------------------- exposure
class ExposureTracker:
    """Accumulates the time spent at each level of each rated indicator.

    Times come from video timestamps (stored videos) or the wall clock (live
    sources). This is a DURATION above exploratory thresholds, not a risk score.
    Gaps longer than max_gap seconds (e.g. dropped frames) are not counted.
    """

    def __init__(self, max_gap=1.0):
        self.max_gap = max_gap
        self.last_t = None
        self.total = 0.0
        self.durations = {n: {lvl: 0.0 for lvl in LEVEL_TEXT} for n in settings.POSTURE_RATED}

    def update(self, assessment, t=None):
        t = time.monotonic() if t is None else t
        if self.last_t is not None:
            dt = t - self.last_t
            if 0 < dt <= self.max_gap:
                self.total += dt
                for n in settings.POSTURE_RATED:
                    lvl = UNAVAILABLE if assessment is None else assessment["indicators"][n]["level"]
                    self.durations[n][lvl] += dt
        self.last_t = t

    def summary(self):
        return {n: {LEVEL_TEXT[l]: round(s, 1) for l, s in d.items()}
                for n, d in self.durations.items()}
