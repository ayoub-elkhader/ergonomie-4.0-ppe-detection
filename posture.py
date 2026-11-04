"""Posture evaluation with MediaPipe Pose.

Measures trunk flexion, neck flexion and knee flexion from the 3D pose
landmarks, then rates each one with thresholds inspired by the RULA method
(Rapid Upper Limb Assessment):

    segment   low risk   medium risk   high risk
    trunk     <= 20 deg  20 - 60 deg   > 60 deg
    neck      <= 10 deg  10 - 20 deg   > 20 deg
    knees     <= 30 deg  30 - 60 deg   > 60 deg   (information only)

The overall verdict uses trunk and neck. Knee flexion is shown for
information: a seated operator naturally has ~90 deg, which is not a risk.
This is an indicator, not a certified ergonomic assessment.
"""
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

LOW, MEDIUM, HIGH = 0, 1, 2
LEVEL_TEXT = {LOW: "OK", MEDIUM: "MODERATE", HIGH: "HIGH RISK"}
LEVEL_BGR = {LOW: (80, 200, 60), MEDIUM: (0, 170, 255), HIGH: (40, 40, 235)}
VERDICT_TEXT = {
    LOW: "Acceptable posture",
    MEDIUM: "Posture to improve",
    HIGH: "Risky posture - correct it now",
}

# segment -> (low limit, high limit) in degrees
THRESHOLDS = {"trunk": (20, 60), "neck": (10, 20), "knees": (30, 60)}
SEGMENT_LABELS = {"trunk": "Trunk flexion", "neck": "Neck flexion", "knees": "Knee flexion"}
VERDICT_SEGMENTS = ("trunk", "neck")


def rate(segment, angle):
    low, high = THRESHOLDS[segment]
    if angle <= low:
        return LOW
    if angle <= high:
        return MEDIUM
    return HIGH


class Ergonomy:
    """Keeps the latest angles of the tracked person."""

    def __init__(self):
        self.detected = False
        self.angles = {"trunk": 0, "neck": 0, "knees": 0}

    # Backwards-compatible accessors
    @property
    def trunk_angle(self):
        return self.angles["trunk"]

    @property
    def neck_angle(self):
        return self.angles["neck"]

    @property
    def leg_angle(self):
        return self.angles["knees"]

    def update_joints(self, landmarks_3d):
        """Update the angles from MediaPipe world landmarks (None = no person)."""
        self.detected = landmarks_3d is not None
        if not self.detected:
            return

        def point(i):
            lm = landmarks_3d.landmark[i]
            return np.array([lm.x, lm.y, lm.z])

        mid_shoulder = (point(11) + point(12)) / 2
        mid_hip = (point(23) + point(24)) / 2
        mid_knee = (point(25) + point(26)) / 2
        mid_ear = (point(7) + point(8)) / 2
        mid_ankle = (point(27) + point(28)) / 2

        self.angles = {
            "trunk": self.get_angle(mid_knee, mid_hip, mid_shoulder, mid_hip),
            "neck": self.get_angle(mid_hip, mid_shoulder, mid_ear, mid_shoulder),
            "knees": self.get_angle(mid_ankle, mid_knee, mid_hip, mid_knee),
        }

    @staticmethod
    def get_angle(a, b, c, d, adjust=True):
        """Angle between vectors (a - b) and (c - d), in degrees.

        With adjust=True the angle is shifted so that a straight posture is 0.
        """
        vec1, vec2 = a - b, c - d
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm == 0:
            return 0
        cosine = np.clip(np.dot(vec1, vec2) / norm, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        return int(abs(angle - 180)) if adjust else int(angle)

    def assessment(self):
        """Return None if nobody is detected, else a dict with per-segment
        ratings and the overall verdict level."""
        if not self.detected:
            return None
        segments = {
            name: {"angle": angle, "level": rate(name, angle)}
            for name, angle in self.angles.items()
        }
        overall = max(segments[s]["level"] for s in VERDICT_SEGMENTS)
        return {"segments": segments, "overall": overall}


def create_pose(static_image=False):
    """MediaPipe Pose estimator (video stream by default, or single images)."""
    return mp_pose.Pose(
        static_image_mode=static_image,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )


def _panel(image, lines, scale):
    """Draw a semi-transparent dark panel with coloured text lines (top-left)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.6 * scale, max(1, int(round(1.6 * scale)))
    pad, gap = int(10 * scale), int(10 * scale)
    sizes = [cv2.getTextSize(text, font, fs, th)[0] for text, _ in lines]
    width = max(w for w, _ in sizes) + 2 * pad
    line_h = max(h for _, h in sizes) + gap
    height = line_h * len(lines) + pad

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, dst=image)
    for i, (text, color) in enumerate(lines):
        y = pad + (i + 1) * line_h - gap // 2
        cv2.putText(image, text, (pad, y), font, fs, color, th, cv2.LINE_AA)


def annotate_posture(frame_bgr, pose, ergonomy):
    """Run pose estimation on a BGR frame, draw the skeleton and a readable
    assessment panel. Returns the annotated image."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)

    image = frame_bgr.copy()
    h, w = image.shape[:2]
    scale = max(0.8, min(w, h) / 500)  # text size follows the frame size

    ergonomy.update_joints(results.pose_world_landmarks)
    result = ergonomy.assessment()

    if results.pose_landmarks is not None:
        thickness = max(2, int(3 * scale))
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec((255, 255, 255), thickness, thickness),
            connection_drawing_spec=mp_drawing.DrawingSpec((255, 200, 0), thickness),
        )

    if result is None:
        _panel(image, [("Posture: no person detected", (220, 220, 220))], scale)
        return image

    lines = [(f"POSTURE: {VERDICT_TEXT[result['overall']]}", LEVEL_BGR[result["overall"]])]
    for name, seg in result["segments"].items():
        info = " (info)" if name not in VERDICT_SEGMENTS else ""
        lines.append((f"{SEGMENT_LABELS[name]}: {seg['angle']} deg - "
                      f"{LEVEL_TEXT[seg['level']]}{info}", LEVEL_BGR[seg["level"]]))
    _panel(image, lines, scale)
    return image


def posture_checker_app(camera_index=0):
    """Stand-alone OpenCV window (press Esc to quit): python posture.py"""
    ergonomy = Ergonomy()
    cap = cv2.VideoCapture(camera_index)
    with create_pose() as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            cv2.imshow("Ergonomie 4.0 - Posture", annotate_posture(frame, pose, ergonomy))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    posture_checker_app()
