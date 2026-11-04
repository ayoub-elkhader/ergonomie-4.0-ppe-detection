"""Model loading, PPE compliance check, posture report and video players."""
from collections import Counter

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

import settings
from posture import (HIGH, LEVEL_TEXT, LOW, MEDIUM, SEGMENT_LABELS, VERDICT_SEGMENTS,
                     VERDICT_TEXT, Ergonomy, annotate_posture, create_pose)

OK, VIOLATION, UNSEEN = "ok", "violation", "unseen"
ICONS = {OK: "✅", VIOLATION: "❌", UNSEEN: "❔"}
LEVEL_ICONS = {LOW: "🟢", MEDIUM: "🟠", HIGH: "🔴"}


@st.cache_resource
def load_model(model_path):
    """Load the YOLOv8 PPE detector once and reuse it across reruns."""
    return YOLO(str(model_path))


@st.cache_data
def load_image(path, max_side=1280):
    """Load an image from disk as an RGB array (st.image can't read Windows paths)."""
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    return np.array(image)


def fit_frame(frame, max_side):
    """Downscale a frame so its longest side is max_side, keeping the aspect ratio."""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


# --------------------------------------------------------------------- PPE
def check_ppe(result):
    """Turn YOLO detections into a PPE checklist.

    Every required item is OK (worn), VIOLATION (missing / badly worn) or
    UNSEEN (not visible in the image, so it cannot be confirmed).
    """
    counts = Counter(result.names[int(c)] for c in result.boxes.cls)
    items = []
    for label, ok_classes, bad_classes in settings.PPE_CHECKLIST:
        bad = [c for c in bad_classes if counts[c]]
        if bad:
            status = VIOLATION
            detail = ", ".join(settings.VIOLATION_LABELS[c] for c in bad)
        elif any(counts[c] for c in ok_classes):
            status, detail = OK, "worn"
        else:
            status, detail = UNSEEN, "not visible - cannot be confirmed"
        items.append({"label": label, "status": status, "detail": detail})

    hazards = [msg for cls, msg in settings.HAZARD_CLASSES.items() if counts[cls]]

    if hazards or any(i["status"] == VIOLATION for i in items):
        overall = VIOLATION
    elif any(i["status"] == UNSEEN for i in items):
        overall = UNSEEN
    else:
        overall = OK
    return {"items": items, "hazards": hazards, "overall": overall, "counts": counts}


def show_ppe_report(container, report, detailed=True):
    """Banner + checklist of the PPE check in a Streamlit container."""
    problems = [f"{i['label']}: {i['detail']}" for i in report["items"]
                if i["status"] == VIOLATION] + report["hazards"]
    unseen = [i["label"] for i in report["items"] if i["status"] == UNSEEN]

    with container.container():
        if report["overall"] == VIOLATION:
            st.error("**PPE NOT COMPLIANT** — " + "; ".join(problems), icon="🚫")
        elif report["overall"] == UNSEEN:
            st.warning("**Check incomplete** — no violation detected, but not visible: "
                       + ", ".join(unseen), icon="⚠️")
        else:
            st.success("**PPE compliant** — all required equipment is worn.", icon="✅")

        if detailed:
            rows = [f"| {ICONS[i['status']]} | **{i['label']}** | {i['detail']} |"
                    for i in report["items"]]
            rows += [f"| ❌ | **Hazard** | {h} |" for h in report["hazards"]]
            st.markdown("| | Equipment | Status |\n|:-:|---|---|\n" + "\n".join(rows))
        else:
            st.markdown("  ".join(f"{ICONS[i['status']]} {i['label']}" for i in report["items"]))


# ----------------------------------------------------------------- Posture
def show_posture_report(container, assessment, detailed=True):
    """Verdict + per-segment angles of the posture assessment."""
    with container.container():
        if assessment is None:
            st.info("**Posture** — no person detected (the whole body should be visible).",
                    icon="🧍")
            return
        overall = assessment["overall"]
        message = f"**Posture: {VERDICT_TEXT[overall]}**"
        {LOW: st.success, MEDIUM: st.warning, HIGH: st.error}[overall](message, icon="🧍")

        segs = assessment["segments"]
        if detailed:
            cols = st.columns(len(segs))
            for col, (name, seg) in zip(cols, segs.items()):
                info = "" if name in VERDICT_SEGMENTS else " (info)"
                col.markdown(
                    f"<div style='text-align:center'>{SEGMENT_LABELS[name]}{info}<br>"
                    f"<span style='font-size:2rem;font-weight:700'>{seg['angle']}°</span><br>"
                    f"{LEVEL_ICONS[seg['level']]} {LEVEL_TEXT[seg['level']]}</div>",
                    unsafe_allow_html=True)
        else:
            st.markdown("  ".join(f"{LEVEL_ICONS[s['level']]} {SEGMENT_LABELS[n]} {s['angle']}°"
                                  for n, s in segs.items()))


# ---------------------------------------------------------- Frame pipeline
class FrameProcessor:
    """Applies PPE detection and/or posture evaluation to frames."""

    def __init__(self, model, conf, use_ppe, use_posture, tracker=None,
                 static_image=False, max_side=720):
        self.model = model
        self.conf = conf
        self.use_ppe = use_ppe
        self.use_posture = use_posture
        self.tracker = tracker
        self.max_side = max_side
        self.pose = create_pose(static_image) if use_posture else None
        self.ergonomy = Ergonomy() if use_posture else None

    def __call__(self, frame):
        """Return (annotated BGR frame, PPE report or None, posture assessment or None)."""
        frame = fit_frame(frame, self.max_side)
        ppe, posture = None, None
        if self.use_ppe:
            if self.tracker:
                res = self.model.track(frame, conf=self.conf, persist=True,
                                       tracker=self.tracker, verbose=False)
            else:
                res = self.model.predict(frame, conf=self.conf, verbose=False)
            ppe = check_ppe(res[0])
            frame = res[0].plot(line_width=max(2, frame.shape[1] // 300))
        if self.use_posture:
            frame = annotate_posture(frame, self.pose, self.ergonomy)
            posture = self.ergonomy.assessment()
        return frame, ppe, posture

    def close(self):
        if self.pose is not None:
            self.pose.close()


def tracker_options():
    if st.sidebar.radio("Display tracker", ("No", "Yes")) == "Yes":
        return st.sidebar.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
    return None


def _run_capture(cap, processor):
    """Read frames until the stream ends; the Stop button interrupts it."""
    col_frame, col_status = st.columns([3, 2])
    frame_slot = col_frame.empty()
    ppe_slot, posture_slot = col_status.empty(), col_status.empty()
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame, ppe, posture = processor(frame)
            frame_slot.image(frame, channels="BGR")
            if processor.use_ppe:
                show_ppe_report(ppe_slot, ppe)
            if processor.use_posture:
                show_posture_report(posture_slot, posture)
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
    if st.sidebar.button("Analyse video", type="primary"):
        st.sidebar.button("Stop")
        _run_capture(cv2.VideoCapture(str(videos[name])), processor_factory(tracker))
    else:
        st.info(f"Video **{name}** selected — click **Analyse video** in the sidebar.")
        st.video(str(videos[name]))


def play_webcam(processor_factory):
    tracker = tracker_options()
    if st.sidebar.button("Start webcam", type="primary"):
        st.sidebar.button("Stop")
        cap = cv2.VideoCapture(settings.WEBCAM_PATH)
        if not cap.isOpened():
            st.error("Unable to open the webcam.")
            return
        _run_capture(cap, processor_factory(tracker))
    else:
        st.info("Click **Start webcam** in the sidebar.")


def play_rtsp_stream(processor_factory):
    url = st.sidebar.text_input("RTSP stream URL")
    st.sidebar.caption("Example: rtsp://user:password@192.168.1.210:554/Streaming/Channels/101")
    tracker = tracker_options()
    if st.sidebar.button("Analyse stream", type="primary") and url:
        st.sidebar.button("Stop")
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            st.error("Unable to open the RTSP stream.")
            return
        _run_capture(cap, processor_factory(tracker))
    else:
        st.info("Enter the camera URL, then click **Analyse stream**.")
