"""Model loading, frame processing and video-source players for the app."""
from collections import Counter

import cv2
import streamlit as st
from ultralytics import YOLO

import settings
from posture import Ergonomy, annotate_posture, create_pose


@st.cache_resource
def load_model(model_path):
    """Load the YOLOv8 PPE detector once and reuse it across reruns."""
    return YOLO(str(model_path))


def summarize(result):
    """Return (Counter of detected class names, list of PPE violations)."""
    names = result.names
    counts = Counter(names[int(c)] for c in result.boxes.cls)
    violations = [
        f"{settings.NON_COMPLIANT_CLASSES[name]} ({n})"
        for name, n in counts.items()
        if name in settings.NON_COMPLIANT_CLASSES
    ]
    return counts, violations


def show_compliance(container, violations, counts):
    """Display a green / red compliance banner in a Streamlit container."""
    if not counts:
        container.info("No PPE element detected.")
    elif violations:
        container.error("Non-compliance detected: " + ", ".join(violations))
    else:
        container.success("PPE compliant: no violation detected.")


def _fit(frame, max_side):
    """Downscale a frame so its longest side is max_side, keeping the aspect ratio."""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


class FrameProcessor:
    """Applies PPE detection and/or posture evaluation to video frames."""

    def __init__(self, model, conf, use_ppe, use_posture, tracker=None):
        self.model = model
        self.conf = conf
        self.use_ppe = use_ppe
        self.use_posture = use_posture
        self.tracker = tracker
        self.pose = create_pose() if use_posture else None
        self.ergonomy = Ergonomy() if use_posture else None

    def __call__(self, frame):
        frame = _fit(frame, max_side=720)
        counts, violations = Counter(), []
        if self.use_ppe:
            if self.tracker:
                res = self.model.track(frame, conf=self.conf, persist=True,
                                       tracker=self.tracker, verbose=False)
            else:
                res = self.model.predict(frame, conf=self.conf, verbose=False)
            counts, violations = summarize(res[0])
            frame = res[0].plot()
        if self.use_posture:
            frame = annotate_posture(frame, self.pose, self.ergonomy)
        return frame, counts, violations

    def close(self):
        if self.pose is not None:
            self.pose.close()


def tracker_options():
    if st.sidebar.radio("Display tracker", ("No", "Yes")) == "Yes":
        return st.sidebar.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
    return None


def _run_capture(cap, processor):
    """Read frames until the stream ends; the Stop button interrupts it."""
    frame_slot, status_slot = st.empty(), st.empty()
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame, counts, violations = processor(frame)
            frame_slot.image(frame, channels="BGR")
            if processor.use_ppe:
                show_compliance(status_slot, violations, counts)
    finally:
        cap.release()
        processor.close()


def play_stored_video(processor_factory):
    videos = settings.list_videos()
    if not videos:
        st.warning(f"No video found. Drop .mp4 files in `{settings.VIDEO_DIR}`.")
        return
    name = st.sidebar.selectbox("Choose a video", list(videos))
    tracker = tracker_options()
    st.video(str(videos[name]))
    if st.sidebar.button("Analyse video"):
        st.sidebar.button("Stop")
        _run_capture(cv2.VideoCapture(str(videos[name])), processor_factory(tracker))


def play_webcam(processor_factory):
    tracker = tracker_options()
    if st.sidebar.button("Start webcam"):
        st.sidebar.button("Stop")
        cap = cv2.VideoCapture(settings.WEBCAM_PATH)
        if not cap.isOpened():
            st.error("Unable to open the webcam.")
            return
        _run_capture(cap, processor_factory(tracker))


def play_rtsp_stream(processor_factory):
    url = st.sidebar.text_input("RTSP stream URL")
    st.sidebar.caption("Example: rtsp://user:password@192.168.1.210:554/Streaming/Channels/101")
    tracker = tracker_options()
    if st.sidebar.button("Analyse stream") and url:
        st.sidebar.button("Stop")
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            st.error("Unable to open the RTSP stream.")
            return
        _run_capture(cap, processor_factory(tracker))
