"""Model loading, frame pipeline, Streamlit reports and video players."""
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

import ppe
import settings
from posture import (HIGH, INDICATORS, LEVEL_TEXT, LOW, OVERALL_TEXT, UNAVAILABLE, ELEVATED,
                     ExposureTracker, PostureEstimator, format_value)

STATUS_ICON = {ppe.WORN: "✅", ppe.MISSING: "❌", ppe.UNCERTAIN: "⚠️", ppe.NOT_OBSERVED: "❔"}
STATUS_TEXT = {ppe.WORN: "worn", ppe.MISSING: "missing / badly worn",
               ppe.UNCERTAIN: "uncertain", ppe.NOT_OBSERVED: "not observed"}
LEVEL_ICON = {LOW: "🟢", ELEVATED: "🟠", HIGH: "🔴", UNAVAILABLE: "⚪"}


# ------------------------------------------------------------------ models
@st.cache_resource
def load_model(model_path):
    """PPE detector shared by all sessions (stateless predict calls only)."""
    return YOLO(str(model_path))


@st.cache_resource
def load_person_model(model_path):
    """COCO detector used only to count people."""
    return YOLO(str(model_path))


def new_model(model_path):
    """Private model instance: object tracking keeps state inside the model,
    so each video run gets its own instance (no state shared between
    sources or browser sessions)."""
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
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------- frame pipeline
class FrameProcessor:
    """PPE detection, people count and posture indicators for one source."""

    def __init__(self, ppe_model, person_model, decision_conf, use_ppe, use_posture,
                 tracker=None, static_image=False, max_side=720):
        self.ppe_model = ppe_model
        self.person_model = person_model
        self.decision_conf = decision_conf
        self.use_ppe = use_ppe
        self.use_posture = use_posture
        self.tracker = tracker
        self.max_side = max_side
        self.posture = PostureEstimator(static_image) if use_posture else None
        self.exposure = ExposureTracker() if use_posture and not static_image else None

    def count_persons(self, frame):
        if self.person_model is None:
            return None
        res = self.person_model.predict(frame, classes=[settings.PERSON_CLASS_ID],
                                        conf=settings.PERSON_CONFIDENCE, verbose=False)[0]
        return ppe.count_persons(res)

    def __call__(self, frame, t=None):
        """Return (annotated BGR frame, PPE report or None, posture assessment or None,
        number of people or None)."""
        frame = fit_frame(frame, self.max_side)
        annotated = frame.copy()
        report, assessment = None, None
        n_persons = self.count_persons(frame)

        if self.use_posture:  # on the original frame, before any drawing
            assessment = self.posture.estimate(frame)
            if self.exposure is not None:
                self.exposure.update(assessment, t)

        if self.use_ppe:
            kwargs = dict(conf=settings.UNCERTAIN_CONFIDENCE, verbose=False)
            res = (self.ppe_model.track(frame, persist=True, tracker=self.tracker, **kwargs)
                   if self.tracker else self.ppe_model.predict(frame, **kwargs))[0]
            report = ppe.check_ppe(ppe.detections_from_result(res, settings.UNCERTAIN_CONFIDENCE),
                                   n_persons, self.decision_conf)
            confirmed = res[res.boxes.conf >= self.decision_conf] if len(res.boxes) else res
            annotated = confirmed.plot(line_width=max(2, frame.shape[1] // 300))

        if self.use_posture:
            self.posture.draw(annotated, assessment)
        return annotated, report, assessment, n_persons

    def close(self):
        if self.posture is not None:
            self.posture.close()


# ------------------------------------------------------------------ reports
def show_ppe_report(container, report):
    """Overall status + checklist."""
    with container.container():
        anomalies = [f"{i['label']}: {i['detail']}" for i in report["items"]
                     if i["status"] == ppe.MISSING] + report["hazards"]
        if report["overall"] == ppe.ANOMALY:
            st.error("**Anomaly detected** — " + "; ".join(anomalies), icon="🚫")
        elif report["overall"] == ppe.INCOMPLETE:
            reasons = report["scope_issues"] + [
                f"{i['label']} {STATUS_TEXT[i['status']]}" for i in report["items"]
                if i["status"] != ppe.WORN]
            reasons += [f"possible {h}" for h in report["uncertain_hazards"]]
            st.warning("**Evaluation incomplete** — " + "; ".join(reasons), icon="⚠️")
        else:
            st.success("**No anomaly detected among the observed items** — the four required "
                       "items were detected for one operator. This is not a certification.",
                       icon="✅")
        if report["scope_issues"] and report["overall"] == ppe.ANOMALY:
            st.caption("Note: " + "; ".join(report["scope_issues"]))

        rows = [f"| {STATUS_ICON[i['status']]} | **{i['label']}** | {STATUS_TEXT[i['status']]} "
                f"| {i['detail']} |" for i in report["items"]]
        rows += [f"| ❌ | **Hazard** | detected | {h} |" for h in report["hazards"]]
        rows += [f"| ⚠️ | **Hazard** | uncertain | {h} |" for h in report["uncertain_hazards"]]
        st.markdown("| | Item | Status | Evidence |\n|:-:|---|---|---|\n" + "\n".join(rows))
        people = "unknown" if report["n_persons"] is None else report["n_persons"]
        st.caption(f"People detected: {people} (the decision supports exactly one operator). "
                   f"Confirmed evidence: detector score ≥ {report['decision_conf']:.2f}; "
                   f"uncertain: {report['uncertain_conf']:.2f}–{report['decision_conf']:.2f}. "
                   "Scores are detector confidences, not probabilities of compliance.")


def show_posture_report(container, assessment, n_persons=None, exposure=None):
    with container.container():
        if assessment is None:
            if n_persons:
                st.info("**Posture** — measurement unavailable: no body pose could be estimated "
                        "(body partly hidden, too small or badly lit).", icon="🧍")
            else:
                st.info("**Posture** — measurement unavailable: no person detected.", icon="🧍")
            return
        overall = assessment["overall"]
        msg = f"**Posture indicators:** {OVERALL_TEXT[overall]}"
        {LOW: st.success, ELEVATED: st.warning, HIGH: st.error,
         UNAVAILABLE: st.info}[overall](msg, icon="🧍")
        if n_persons is not None and n_persons > 1:
            st.caption(f"{n_persons} people detected: MediaPipe measures only one of them.")

        rows = []
        for name, label in INDICATORS.items():
            ind = assessment["indicators"][name]
            role = "rated" if name in settings.POSTURE_RATED else "info"
            asym = " (asymmetry)" if ind["asymmetry"] else ""
            rows.append(f"| {LEVEL_ICON[ind['level']]} | {label} | {format_value(ind)}{asym} "
                        f"| {LEVEL_TEXT[ind['level']]} | {role} |")
        st.markdown("| | Indicator | Value | Level | Use |\n|:-:|---|---|---|---|\n" + "\n".join(rows))

        if exposure is not None and exposure.total > 0:
            parts = []
            for name in settings.POSTURE_RATED:
                d = exposure.summary()[name]
                parts.append(f"{INDICATORS[name]}: elevated {d['elevated']} s, high {d['high']} s, "
                             f"unavailable {d['unavailable']} s")
            st.caption(f"Duration over {exposure.total:.1f} s analysed — " + " · ".join(parts)
                       + ". Durations above exploratory thresholds, not a risk score.")
        st.caption("Exploratory indicators from a single camera (see README): not a RULA/REBA "
                   "score. Trunk inclination assumes a level camera and a side view.")


# ------------------------------------------------------------ video sources
def tracker_options():
    if st.sidebar.radio("Display tracker", ("No", "Yes")) == "Yes":
        return st.sidebar.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
    return None


def _run_capture(cap, processor, use_video_clock):
    """Read frames until the stream ends. The Stop button (or any widget
    change) triggers a Streamlit rerun, which interrupts this loop; the
    finally block then releases the capture and MediaPipe."""
    col_frame, col_status = st.columns([3, 2])
    frame_slot = col_frame.empty()
    fps_slot = col_frame.empty()
    ppe_slot, posture_slot = col_status.empty(), col_status.empty()
    n, start = 0, time.perf_counter()
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if use_video_clock else None
            frame, report, assessment, n_persons = processor(frame, t)
            n += 1
            frame_slot.image(frame, channels="BGR")
            fps_slot.caption(f"Processing speed: {n / (time.perf_counter() - start):.1f} frames/s")
            if report is not None:
                show_ppe_report(ppe_slot, report)
            if processor.use_posture:
                show_posture_report(posture_slot, assessment, n_persons, processor.exposure)
        if n == 0:
            st.error("No frame could be read from this source.")
    except Exception as ex:  # noqa: BLE001 - report the failure in the UI
        st.error(f"Processing stopped: {ex}")
    finally:
        cap.release()
        processor.close()


def play_stored_video(processor_factory):
    videos = settings.list_videos()
    if not videos:
        st.warning(f"No video found. Drop .mp4 files in the `videos/` folder.")
        return
    name = st.sidebar.selectbox("Choose a video", list(videos))
    tracker = tracker_options()
    path = videos[name]
    if st.sidebar.button("Analyse video", type="primary"):
        st.sidebar.button("Stop")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            st.error(f"Unable to open the video `{path.name}` (missing codec or corrupted file).")
            return
        _run_capture(cap, processor_factory(tracker), use_video_clock=True)
    else:
        st.info(f"Video **{name}** selected — click **Analyse video** in the sidebar.")
        st.video(path.read_bytes())


def play_webcam(processor_factory):
    tracker = tracker_options()
    if st.sidebar.button("Start webcam", type="primary"):
        st.sidebar.button("Stop")
        cap = cv2.VideoCapture(settings.WEBCAM_PATH)
        if not cap.isOpened():
            st.error("Unable to open the webcam (in use by another program, or no camera).")
            return
        _run_capture(cap, processor_factory(tracker), use_video_clock=False)
    else:
        st.info("Click **Start webcam** in the sidebar.")


def play_rtsp_stream(processor_factory):
    url = st.sidebar.text_input("RTSP stream URL")
    st.sidebar.caption("Example: rtsp://user:password@192.168.1.210:554/Streaming/Channels/101")
    tracker = tracker_options()
    if st.sidebar.button("Analyse stream", type="primary"):
        if not url.strip().lower().startswith("rtsp://"):
            st.error("Enter a URL starting with rtsp://")
            return
        st.sidebar.button("Stop")
        cap = cv2.VideoCapture(url.strip())
        if not cap.isOpened():
            st.error("Unable to open the RTSP stream (check the URL, credentials and network).")
            return
        _run_capture(cap, processor_factory(tracker), use_video_clock=False)
    else:
        st.info("Enter the camera URL, then click **Analyse stream**.")
