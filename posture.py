"""Posture evaluation with MediaPipe Pose.

Computes trunk, neck and leg angles from the 3D pose landmarks and colours
them green / yellow / red according to ergonomic thresholds.
"""
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

GREEN, YELLOW, RED = (0, 255, 0), (0, 255, 255), (0, 0, 255)  # BGR

# (green below, yellow up to, red above) in degrees
THRESHOLDS = {
    "trunk": (20, 60),
    "neck": (20, 30),
    "leg": (30, 50),
}


class Ergonomy:
    """Keeps the latest trunk / neck / leg angles of the tracked person."""

    def __init__(self):
        self.trunk_angle = 0
        self.neck_angle = 0
        self.leg_angle = 0

    def update_joints(self, landmarks_3d):
        """Update the angles from MediaPipe world landmarks (None = no person)."""
        if landmarks_3d is None:
            return

        def point(i):
            lm = landmarks_3d.landmark[i]
            return np.array([lm.x, lm.y, lm.z])

        mid_shoulder = (point(11) + point(12)) / 2
        mid_hip = (point(23) + point(24)) / 2
        mid_knee = (point(25) + point(26)) / 2
        mid_ear = (point(7) + point(8)) / 2
        mid_ankle = (point(27) + point(28)) / 2

        self.trunk_angle = self.get_angle(mid_knee, mid_hip, mid_shoulder, mid_hip)
        self.neck_angle = self.get_angle(mid_hip, mid_shoulder, mid_ear, mid_shoulder)
        self.leg_angle = self.get_angle(mid_ankle, mid_knee, mid_hip, mid_knee)

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

    @staticmethod
    def _color(angle, limits):
        low, high = limits
        if angle < low:
            return GREEN
        if angle <= high:
            return YELLOW
        return RED

    def get_trunk_color(self):
        return self._color(self.trunk_angle, THRESHOLDS["trunk"])

    def get_neck_color(self):
        return self._color(self.neck_angle, THRESHOLDS["neck"])

    def get_leg_color(self):
        return self._color(self.leg_angle, THRESHOLDS["leg"])


def create_pose(static_image=False):
    """MediaPipe Pose estimator (video stream by default, or single images)."""
    return mp_pose.Pose(
        static_image_mode=static_image,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )


def annotate_posture(frame_bgr, pose, ergonomy):
    """Run pose estimation on a BGR frame and draw skeleton + angle bars."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)

    image = frame_bgr.copy()
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    ergonomy.update_joints(results.pose_world_landmarks)

    rows = [
        ("trunk", ergonomy.trunk_angle, ergonomy.get_trunk_color()),
        ("neck", ergonomy.neck_angle, ergonomy.get_neck_color()),
        ("leg", ergonomy.leg_angle, ergonomy.get_leg_color()),
    ]
    # Bars shrink as the angle grows; sized to fit portrait frames too.
    bar_width = 180
    for i, (name, angle, color) in enumerate(rows):
        top = 5 + i * 28
        bar = max(0, bar_width - angle * 2)
        cv2.rectangle(image, (5, top), (5 + bar_width, top + 20), (255, 255, 255), -1)
        cv2.rectangle(image, (5, top), (5 + bar, top + 20), color, -1)
        cv2.putText(image, f"{name}: {angle} deg", (bar_width + 12, top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
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
