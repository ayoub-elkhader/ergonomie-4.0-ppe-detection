"""Ergonomie 4.0 - PPE compliance and posture monitoring for welding workshops.

Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import helper
import settings
from posture import Ergonomy, annotate_posture, create_pose

st.set_page_config(
    page_title="Ergonomie 4.0 - PPE & posture detection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Ergonomie 4.0 — PPE & posture detection in a welding workshop")
st.caption("YOLOv8 detects personal protective equipment, MediaPipe evaluates the operator's posture.")

# ------------------------------------------------------------------ Sidebar
st.sidebar.header("Analysis")
analysis = st.sidebar.radio("What to analyse", ("PPE detection", "Posture", "PPE + posture"))
use_ppe = analysis != "Posture"
use_posture = analysis != "PPE detection"

confidence = st.sidebar.slider("Model confidence", 25, 100, 40) / 100

model = None
if use_ppe:
    try:
        model = helper.load_model(settings.DETECTION_MODEL)
    except Exception as ex:  # noqa: BLE001 - show any loading error to the user
        st.error(f"Unable to load the model at {settings.DETECTION_MODEL}: {ex}")
        st.stop()

st.sidebar.header("Source")
source = st.sidebar.radio("Select source", settings.SOURCES_LIST)


def processor_factory(tracker=None):
    return helper.FrameProcessor(model, confidence, use_ppe, use_posture, tracker)


# -------------------------------------------------------------------- Image
if source == settings.IMAGE:
    uploaded = st.sidebar.file_uploader("Choose an image", type=("jpg", "jpeg", "png", "bmp", "webp"))
    col1, col2 = st.columns(2)

    if uploaded is None:
        col1.image(str(settings.DEFAULT_IMAGE), caption="Example image")
        col2.image(str(settings.DEFAULT_DETECT_IMAGE), caption="Example detection")
        st.info("Upload an image in the sidebar, then click **Analyse image**.")
    else:
        image = Image.open(uploaded).convert("RGB")
        col1.image(image, caption="Uploaded image")

        if st.sidebar.button("Analyse image"):
            frame = np.array(image)[:, :, ::-1].copy()  # RGB -> BGR
            counts, violations = {}, []
            if use_ppe:
                result = model.predict(frame, conf=confidence, verbose=False)[0]
                counts, violations = helper.summarize(result)
                frame = result.plot()
            if use_posture:
                with create_pose(static_image=True) as pose:
                    frame = annotate_posture(frame, pose, Ergonomy())
            col2.image(frame, channels="BGR", caption="Result")

            if use_ppe:
                helper.show_compliance(st, violations, counts)
                if counts:
                    st.dataframe(
                        pd.DataFrame(sorted(counts.items()), columns=["Class", "Count"]),
                        hide_index=True,
                    )

# ------------------------------------------------------------ Video sources
elif source == settings.VIDEO:
    helper.play_stored_video(processor_factory)

elif source == settings.WEBCAM:
    helper.play_webcam(processor_factory)

elif source == settings.RTSP:
    helper.play_rtsp_stream(processor_factory)
