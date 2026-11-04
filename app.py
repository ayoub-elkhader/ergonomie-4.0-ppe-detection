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
st.caption("PPE compliance and posture monitoring for welding workshops — "
           "YOLOv8 detects protective equipment, MediaPipe evaluates the operator's posture.")

# ------------------------------------------------------------------ Sidebar
st.sidebar.header("⚙️ Analysis")
analysis = st.sidebar.radio("What to analyse", ("PPE detection", "Posture", "PPE + posture"))
use_ppe = analysis != "Posture"
use_posture = analysis != "PPE detection"

confidence = st.sidebar.slider("Detection confidence", 25, 100, 50,
                               help="Minimum confidence for a PPE detection to be kept.") / 100

model = None
if use_ppe:
    try:
        model = helper.load_model(settings.DETECTION_MODEL)
    except Exception as ex:  # noqa: BLE001 - show any loading error to the user
        st.error(f"Unable to load the model at {settings.DETECTION_MODEL}: {ex}")
        st.stop()

st.sidebar.header("📷 Source")
source = st.sidebar.radio("Select source", settings.SOURCES_LIST)


def processor_factory(tracker=None, static_image=False, max_side=720):
    return helper.FrameProcessor(model, confidence, use_ppe, use_posture, tracker,
                                 static_image=static_image, max_side=max_side)


def show_welcome():
    """Landing page shown before any image is uploaded."""
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("#### 🦺 PPE detection\nHelmet, gloves, apron, safety shoes, "
                    "plus hazards such as loose hair or jewellery.")
    with c2.container(border=True):
        st.markdown("#### ✅ Compliance check\nEach required item is reported as "
                    "**worn**, **missing** or **not visible**.")
    with c3.container(border=True):
        st.markdown("#### 🧍 Posture\nTrunk, neck and knee angles rated with "
                    "RULA-inspired thresholds.")

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
        image = Image.open(uploaded).convert("RGB")
        image.thumbnail((1280, 1280))
        analyse = st.sidebar.button("Analyse image", type="primary")

        col1, col2 = st.columns(2)
        col1.image(image, caption="Uploaded image", use_column_width=True)

        if not analyse:
            col2.info("Click **Analyse image** in the sidebar.", icon="👈")
        else:
            processor = processor_factory(static_image=True, max_side=1280)
            frame, ppe, posture = processor(np.array(image)[:, :, ::-1].copy())
            processor.close()
            col2.image(frame, channels="BGR", caption="Result", use_column_width=True)

            reports = [r for r in (use_ppe, use_posture) if r]
            cols = st.columns(len(reports))
            i = 0
            if use_ppe:
                cols[i].subheader("PPE compliance")
                helper.show_ppe_report(cols[i], ppe)
                i += 1
            if use_posture:
                cols[i].subheader("Posture")
                helper.show_posture_report(cols[i], posture)

# ------------------------------------------------------------ Video sources
elif source == settings.VIDEO:
    helper.play_stored_video(processor_factory)

elif source == settings.WEBCAM:
    helper.play_webcam(processor_factory)

elif source == settings.RTSP:
    helper.play_rtsp_stream(processor_factory)
