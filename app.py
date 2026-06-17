import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="DeepFake Forensics Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0F0F1A; }
    .stApp { background-color: #0F0F1A; }
    .metric-card {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border-radius: 16px; padding: 24px;
        border: 1px solid #2D2D44; text-align: center;
    }
    .score-big { font-size: 3em; font-weight: 800; }
    .verdict-real { color: #00D4AA; }
    .verdict-fake { color: #FF4757; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    from inference import DeepfakeDetector
    return DeepfakeDetector()


def render_gauge_chart(score, title="Forgery Score"):
    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        title={"text": title, "font": {"size": 18, "color": "#E8E8E8"}},
        number={"suffix": "%", "font": {"color": "#E8E8E8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#E8E8E8"},
            "bar": {"color": "#FF4757" if score > 0.5 else "#00D4AA"},
            "bgcolor": "#1A1A2E",
            "steps": [
                {"range": [0, 30], "color": "#00D4AA33"},
                {"range": [30, 70], "color": "#FFA50233"},
                {"range": [70, 100], "color": "#FF475733"},
            ],
            "threshold": {
                "line": {"color": "#FFA502", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#1A1A2E",
        font={"color": "#E8E8E8"},
        height=300,
        margin=dict(t=60, b=20, l=30, r=30),
    )
    return fig


def render_radar_chart(scores):
    import plotly.graph_objects as go

    categories = list(scores.keys())
    values = list(scores.values())
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        fillcolor="rgba(91, 141, 239, 0.2)",
        line=dict(color="#5B8DEF", width=2),
        name="Scores",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1A1A2E",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2D2D44"),
            angularaxis=dict(gridcolor="#2D2D44"),
        ),
        paper_bgcolor="#1A1A2E",
        font=dict(color="#E8E8E8"),
        height=400,
        margin=dict(t=40, b=40, l=80, r=80),
        showlegend=False,
    )
    return fig


def render_timeline(frame_scores):
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = ["#FF4757" if s > 0.5 else "#00D4AA" for s in frame_scores]

    fig.add_trace(go.Scatter(
        y=frame_scores, mode="lines+markers",
        line=dict(color="#5B8DEF", width=2),
        marker=dict(color=colors, size=6),
        name="Fake Probability",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#FFA502", annotation_text="Threshold")
    fig.update_layout(
        paper_bgcolor="#1A1A2E",
        plot_bgcolor="#1A1A2E",
        font=dict(color="#E8E8E8"),
        xaxis=dict(title="Frame Sample", gridcolor="#2D2D44"),
        yaxis=dict(title="Fake Probability", range=[0, 1], gridcolor="#2D2D44"),
        height=350,
        margin=dict(t=30, b=50, l=50, r=30),
    )
    return fig


st.title("🔬 DeepFake Forensics Analyzer")
st.markdown("*Multi-modal deepfake detection with frequency analysis, facial forensics, and temporal consistency checking.*")

with st.sidebar:
    st.header("⚙️ Settings")
    analysis_mode = st.selectbox("Analysis Mode", ["Image", "Video"])
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    show_frequency = st.checkbox("Show Frequency Analysis", True)
    show_facial = st.checkbox("Show Facial Forensics", True)
    show_heatmap = st.checkbox("Show Attention Heatmap", True)

    st.header("📊 About")
    st.markdown("""
    **Models Used:**
    - EfficientNet-B4 + SRM Filters
    - XceptionNet
    - Capsule Network
    - Frequency-Aware Network

    **Analysis Methods:**
    - FFT / DCT / Wavelet
    - Facial Landmark Consistency
    - Skin Texture Analysis
    - Blending Artifact Detection
    - Optical Flow Consistency
    """)

if analysis_mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded is not None:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Running forensic analysis..."):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    image.save(tmp.name)
                    try:
                        detector = load_detector()
                        result = detector.analyze_image(tmp.name, generate_report=False)
                    finally:
                        os.unlink(tmp.name)

            scores = result["scores"]
            verdict = result["verdict"]

            st.markdown(
                f'<div class="metric-card"><div class="score-big verdict-{"fake" if verdict == "FAKE" else "real"}">'
                f'{verdict}</div><p>{scores["overall_score"]:.1%} forgery probability</p></div>',
                unsafe_allow_html=True,
            )

        st.divider()

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Overall Score", f"{scores['overall_score']:.1%}")
        with col_b:
            st.metric("Neural Network", f"{scores['neural_score']:.1%}")
        with col_c:
            st.metric("Frequency Analysis", f"{scores['frequency_score']:.1%}")
        with col_d:
            st.metric("Facial Forensics", f"{scores['facial_score']:.1%}")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(render_gauge_chart(scores["overall_score"]), use_container_width=True)
        with col_g2:
            radar_scores = {k.replace("_score", "").title(): v for k, v in scores.items() if k != "overall_score"}
            st.plotly_chart(render_radar_chart(radar_scores), use_container_width=True)

        if show_frequency and "frequency_analysis" in result:
            with st.expander("📈 Frequency Domain Analysis", expanded=True):
                freq = result["frequency_analysis"]
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.markdown("**FFT Magnitude Spectrum**")
                    mag = freq["fft"]["magnitude_spectrum"]
                    mag_normalized = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255).astype(np.uint8)
                    st.image(mag_normalized, caption="Frequency magnitude", use_container_width=True)
                with col_f2:
                    st.markdown("**Anomaly Indicators**")
                    for name, val in freq["anomaly_indicators"].items():
                        label = name.replace("_", " ").title()
                        st.progress(min(float(val), 1.0), text=f"{label}: {val:.3f}")

        if show_facial and "facial_analysis" in result:
            with st.expander("👤 Facial Forensics", expanded=True):
                facial = result["facial_analysis"]
                col_fa, col_fb = st.columns(2)
                with col_fa:
                    st.json({
                        "Symmetry": f"{facial['consistency'].get('symmetry_score', 0):.3f}",
                        "Jaw Smoothness": f"{facial['consistency'].get('jaw_smoothness', 0):.4f}",
                        "Texture Variance": f"{facial['texture'].get('texture_variance', 0):.1f}",
                    })
                with col_fb:
                    st.json({
                        "Blending Score": f"{facial['blending'].get('blending_score', 0):.3f}",
                        "Color Consistency": f"{facial['texture'].get('color_consistency', 0):.3f}",
                        "Edge Density": f"{facial['blending'].get('edge_density_at_boundary', 0):.4f}",
                    })

else:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(uploaded)

        with st.spinner("Analyzing video frames... This may take a moment."):
            try:
                detector = load_detector()
                result = detector.analyze_video(tmp_path, generate_report=False)
            finally:
                os.unlink(tmp_path)

        scores = result["scores"]
        verdict = result["verdict"]

        st.markdown(
            f'<div class="metric-card"><div class="score-big verdict-{"fake" if verdict == "FAKE" else "real"}">'
            f'{verdict}</div><p>{scores["overall_score"]:.1%} forgery probability</p></div>',
            unsafe_allow_html=True,
        )

        st.divider()

        cols = st.columns(4)
        labels = ["Overall", "Neural Network", "Frequency", "Temporal"]
        keys = ["overall_score", "neural_score", "frequency_score", "temporal_score"]
        for col, label, key in zip(cols, labels, keys):
            with col:
                st.metric(label, f"{scores.get(key, 0):.1%}")

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            st.plotly_chart(render_gauge_chart(scores["overall_score"]), use_container_width=True)
        with col_v2:
            if result.get("frame_scores"):
                st.plotly_chart(render_timeline(result["frame_scores"]), use_container_width=True)

        radar_scores = {k.replace("_score", "").title(): v for k, v in scores.items() if k != "overall_score"}
        st.plotly_chart(render_radar_chart(radar_scores), use_container_width=True)

st.divider()
st.markdown(
    '<p style="text-align:center; color:#555;">DeepFake Forensics Analyzer v1.0.0 — '
    'Built with PyTorch, OpenCV, and Streamlit</p>',
    unsafe_allow_html=True,
)
