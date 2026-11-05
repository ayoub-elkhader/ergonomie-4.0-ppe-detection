"""Ergonomie 4.0 - PPE compliance and posture monitoring for welding workshops.

Run with:  streamlit run app.py
"""
import numpy as np
import streamlit as st
from PIL import Image

import helper
import settings

st.set_page_config(
    page_title="Ergonomie 4.0 - PPE & posture detection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 2rem;}
      h1 {font-size: 2.2rem !important; margin-bottom: 0 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🦺 Ergonomie 4.0")
st.caption("Prototype for PPE detection and exploratory posture indicators in a welding "
           "workshop — YOLOv8 detects protective equipment, MediaPipe estimates the pose. "
           "Decision support only: not a certified safety or ergonomic assessment.")

# ------------------------------------------------------------------ Sidebar
st.sidebar.header("⚙️ Analysis")
analysis = st.sidebar.radio("What to analyse", ("PPE detection", "Posture", "PPE + posture"))
use_ppe = analysis != "Posture"
use_posture = analysis != "PPE detection"

confidence = st.sidebar.slider(
    "Decision threshold (detector score)", 30, 95,
    int(settings.DEFAULT_DECISION_CONFIDENCE * 100), step=5,
    help="Detections at or above this score count as confirmed evidence; detections between "
         f"{settings.UNCERTAIN_CONFIDENCE:.2f} and this value are reported as uncertain. "
         "A detector score is not a probability of compliance.") / 100


def load_models():
    try:
        ppe_model = helper.load_model(settings.DETECTION_MODEL) if use_ppe else None
    except Exception as ex:  # noqa: BLE001 - show any loading error to the user
        st.error(f"Unable to load the PPE model `{settings.DETECTION_MODEL.name}`: {ex}")
        st.stop()
    try:
        person_model = helper.load_person_model(settings.PERSON_MODEL)
    except Exception as ex:  # noqa: BLE001
        st.warning(f"Person counter unavailable ({ex}): the number of operators cannot be "
                   "checked, so PPE evaluations will stay incomplete.")
        person_model = None
    return ppe_model, person_model


PPE_MODEL, PERSON_MODEL = load_models()

st.sidebar.header("📷 Source")
source = st.sidebar.radio("Select source", settings.SOURCES_LIST)


def processor_factory(tracker=None, static_image=False, max_side=720):
    # Tracking stores state inside the model: give each run its own instance.
    ppe_model = helper.new_model(settings.DETECTION_MODEL) if (use_ppe and tracker) else PPE_MODEL
    return helper.FrameProcessor(ppe_model, PERSON_MODEL, confidence, use_ppe, use_posture,
                                 tracker, static_image=static_image, max_side=max_side)


def show_welcome():
    """Landing page shown before any image is uploaded."""
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("#### 🦺 PPE detection\nHelmet, gloves, apron, safety shoes, "
                    "plus hazards such as uncovered hair or jewellery.")
    with c2.container(border=True):
        st.markdown("#### ✅ Checklist for one operator\nEach required item is **worn**, "
                    "**missing**, **uncertain** or **not observed**. Nothing is assumed "
                    "from a missing detection.")
    with c3.container(border=True):
        st.markdown("#### 🧍 Posture indicators\nTrunk inclination, neck, hip and knee "
                    "flexion with exploratory, configurable thresholds.")

    st.subheader("Example")
    left, right = st.columns(2)
    left.image(helper.load_image(settings.DEFAULT_IMAGE), caption="Input image",
               use_column_width=True)
    right.image(helper.load_image(settings.DEFAULT_DETECT_IMAGE), caption="Detected PPE",
                use_column_width=True)
    st.info("**How to use:** choose the analysis and the source in the sidebar, "
            "upload an image, then click **Analyse image**.", icon="👈")


# -------------------------------------------------------------------- Image
if source == settings.IMAGE:
    uploaded = st.sidebar.file_uploader("Choose an image", type=("jpg", "jpeg", "png", "bmp", "webp"))

    if uploaded is None:
        show_welcome()
    else:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception as ex:  # noqa: BLE001 - corrupted or unsupported file
            st.error(f"Unable to read this image: {ex}")
            st.stop()
        image.thumbnail((1280, 1280))
        analyse = st.sidebar.button("Analyse image", type="primary")

        col1, col2 = st.columns(2)
        col1.image(image, caption="Uploaded image", use_column_width=True)

        if not analyse:
            col2.info("Click **Analyse image** in the sidebar.", icon="👈")
        else:
            processor = processor_factory(static_image=True, max_side=1280)
            try:
                frame, report, assessment, n_persons = processor(
                    np.array(image)[:, :, ::-1].copy())
            finally:
                processor.close()
            col2.image(frame, channels="BGR", caption="Result (confirmed detections only)",
                       use_column_width=True)

            cols = st.columns(int(use_ppe) + int(use_posture))
            i = 0
            if use_ppe:
                cols[i].subheader("PPE checklist")
                helper.show_ppe_report(cols[i], report)
                i += 1
            if use_posture:
                cols[i].subheader("Posture indicators")
                helper.show_posture_report(cols[i], assessment, n_persons)

# ------------------------------------------------------------ Video sources
elif source == settings.VIDEO:
    helper.play_stored_video(processor_factory)

elif source == settings.WEBCAM:
    helper.play_webcam(processor_factory)

elif source == settings.RTSP:
    helper.play_rtsp_stream(processor_factory)
