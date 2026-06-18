import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import glob
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
    .metric-card {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        border-radius: 16px; padding: 24px;
        border: 1px solid #2D2D44; text-align: center;
        margin-bottom: 20px;
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
        mode="gauge+number",
        value=score * 100,
        title={"text": title, "font": {"size": 18}},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#FF4757" if score > 0.5 else "#00D4AA"},
            "steps": [
                {"range": [0, 30], "color": "#e8f5e9"},
                {"range": [30, 70], "color": "#fff3e0"},
                {"range": [70, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "#FFA502", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
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
        fillcolor="rgba(91, 141, 239, 0.3)",
        line=dict(color="#5B8DEF", width=2),
        name="Scores",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400,
        margin=dict(t=40, b=40, l=80, r=80),
        showlegend=False,
    )
    return fig


def render_bar_chart(scores):
    import plotly.graph_objects as go

    names = [k.replace("_score", "").replace("_", " ").title() for k in scores.keys()]
    values = list(scores.values())
    colors = ["#FF4757" if v > 0.5 else "#00D4AA" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors, text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#FFA502", line_width=2)
    fig.update_layout(
        xaxis=dict(range=[0, 1], title="Score"),
        height=250,
        margin=dict(t=20, b=40, l=120, r=40),
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
        xaxis=dict(title="Frame Sample"),
        yaxis=dict(title="Fake Probability", range=[0, 1]),
        height=350,
        margin=dict(t=30, b=50, l=50, r=30),
    )
    return fig


def get_sample_images():
    samples = {"fake": [], "real": []}
    for label in ["fake", "real"]:
        folder = f"data/processed/test/{label}"
        if os.path.isdir(folder):
            files = sorted(glob.glob(os.path.join(folder, "*")))[:5]
            samples[label] = files
    return samples


st.title("🔬 DeepFake Forensics Analyzer")
st.markdown("*Multi-modal deepfake detection with frequency analysis, facial forensics, and temporal consistency checking.*")

with st.sidebar:
    st.header("⚙️ Settings")
    analysis_mode = st.selectbox("Analysis Mode", ["Image", "Video"])
    show_frequency = st.checkbox("Show Frequency Analysis", True)
    show_facial = st.checkbox("Show Facial Forensics", True)

    st.divider()
    st.header("📊 About")
    st.markdown("""
    **Models:** EfficientNet-B4 + SRM Filters

    **Analysis Methods:**
    - FFT / DCT / Wavelet
    - Facial Landmark Consistency
    - Skin Texture Analysis
    - Blending Artifact Detection
    - Optical Flow (Video)
    """)

if analysis_mode == "Image":
    st.subheader("📁 Choose an image")

    input_method = st.radio(
        "How to provide image:",
        ["Upload a file", "Use a sample from dataset"],
        horizontal=True,
    )

    image_path = None

    if input_method == "Upload a file":
        uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(uploaded.getvalue())
            tmp.close()
            image_path = tmp.name

    else:
        samples = get_sample_images()
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**🟢 Real samples:**")
            for path in samples["real"]:
                if st.button(f"📷 {os.path.basename(path)}", key=f"real_{path}"):
                    st.session_state["selected_sample"] = path
        with col_s2:
            st.markdown("**🔴 Fake samples:**")
            for path in samples["fake"]:
                if st.button(f"📷 {os.path.basename(path)}", key=f"fake_{path}"):
                    st.session_state["selected_sample"] = path

        if "selected_sample" in st.session_state:
            image_path = st.session_state["selected_sample"]

    if image_path is not None:
        st.divider()

        image = Image.open(image_path)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Verdict")
            with st.spinner("🔍 Running forensic analysis..."):
                detector = load_detector()
                result = detector.analyze_image(image_path, generate_report=True)

            scores = result["scores"]
            verdict = result["verdict"]

            verdict_class = "fake" if verdict == "FAKE" else "real"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="score-big verdict-{verdict_class}">{verdict}</div>'
                f'<p style="font-size:1.3em;">{scores["overall_score"]:.1%} forgery probability</p>'
                f'<p>Confidence: {result["confidence"]:.1%}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if input_method == "Upload a file":
            os.unlink(image_path)

        st.divider()
        st.subheader("📊 Score Breakdown")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("🎯 Overall", f"{scores['overall_score']:.1%}")
        with col_b:
            st.metric("🧠 Neural Net", f"{scores['neural_score']:.1%}")
        with col_c:
            st.metric("📡 Frequency", f"{scores['frequency_score']:.1%}")
        with col_d:
            st.metric("👤 Facial", f"{scores['facial_score']:.1%}")

        st.divider()
        st.subheader("📈 Visual Analysis")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(render_gauge_chart(scores["overall_score"], "Overall Forgery Score"), use_container_width=True)
        with col_g2:
            radar_scores = {k.replace("_score", "").replace("_", " ").title(): v for k, v in scores.items() if k != "overall_score"}
            st.plotly_chart(render_radar_chart(radar_scores), use_container_width=True)

        st.plotly_chart(render_bar_chart({k: v for k, v in scores.items() if k != "overall_score"}), use_container_width=True)

        if show_frequency and "frequency_analysis" in result:
            st.divider()
            st.subheader("📡 Frequency Domain Analysis")
            freq = result["frequency_analysis"]

            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                st.markdown("**FFT Magnitude Spectrum**")
                mag = freq["fft"]["magnitude_spectrum"]
                mag_norm = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255).astype(np.uint8)
                mag_color = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
                mag_color = cv2.cvtColor(mag_color, cv2.COLOR_BGR2RGB)
                st.image(mag_color, caption="FFT Magnitude", use_container_width=True)

            with col_f2:
                st.markdown("**FFT Phase Spectrum**")
                phase = freq["fft"]["phase_spectrum"]
                phase_norm = ((phase - phase.min()) / (phase.max() - phase.min() + 1e-8) * 255).astype(np.uint8)
                phase_color = cv2.applyColorMap(phase_norm, cv2.COLORMAP_TWILIGHT)
                phase_color = cv2.cvtColor(phase_color, cv2.COLOR_BGR2RGB)
                st.image(phase_color, caption="FFT Phase", use_container_width=True)

            with col_f3:
                st.markdown("**DCT Energy Map**")
                dct = freq["dct"]["dct_energy_map"]
                dct_img = np.log1p(dct)
                dct_norm = ((dct_img - dct_img.min()) / (dct_img.max() - dct_img.min() + 1e-8) * 255).astype(np.uint8)
                dct_resized = cv2.resize(dct_norm, (256, 256), interpolation=cv2.INTER_NEAREST)
                dct_color = cv2.applyColorMap(dct_resized, cv2.COLORMAP_MAGMA)
                dct_color = cv2.cvtColor(dct_color, cv2.COLOR_BGR2RGB)
                st.image(dct_color, caption="DCT Energy", use_container_width=True)

            st.markdown("**Anomaly Indicators**")
            for name, val in freq["anomaly_indicators"].items():
                label = name.replace("_", " ").title()
                col_ind, col_bar = st.columns([1, 3])
                with col_ind:
                    st.write(f"**{label}**")
                with col_bar:
                    st.progress(min(float(val), 1.0), text=f"{val:.3f}")

            st.markdown(f"**Spectral Slope:** {freq['spectral_slope']['slope']:.3f} "
                        f"(R² = {freq['spectral_slope']['r_squared']:.3f})")
            st.markdown(f"**Wavelet Detail-to-Approx Ratio:** {freq['wavelet']['detail_to_approx_ratio']:.3f}")
            st.markdown(f"**DCT High-Freq Ratio:** {freq['dct']['high_freq_ratio']:.4f}")

        if show_facial and "facial_analysis" in result:
            st.divider()
            st.subheader("👤 Facial Forensics")
            facial = result["facial_analysis"]

            col_fa, col_fb, col_fc = st.columns(3)
            with col_fa:
                st.markdown("**Landmark Consistency**")
                st.metric("Symmetry", f"{facial['consistency'].get('symmetry_score', 0):.3f}")
                st.metric("Proportion Dev.", f"{facial['consistency'].get('proportion_deviation', 0):.4f}")
                st.metric("Jaw Smoothness", f"{facial['consistency'].get('jaw_smoothness', 0):.4f}")
            with col_fb:
                st.markdown("**Skin Texture**")
                st.metric("Texture Variance", f"{facial['texture'].get('texture_variance', 0):.1f}")
                st.metric("Gabor Uniformity", f"{facial['texture'].get('gabor_uniformity', 0):.3f}")
                st.metric("Color Consistency", f"{facial['texture'].get('color_consistency', 0):.3f}")
            with col_fc:
                st.markdown("**Blending Artifacts**")
                st.metric("Blending Score", f"{facial['blending'].get('blending_score', 0):.3f}")
                st.metric("Edge Density", f"{facial['blending'].get('edge_density_at_boundary', 0):.4f}")
                st.metric("Color Discontinuity", f"{facial['blending'].get('color_discontinuity', 0):.4f}")

        fig_dir = "outputs/figures"
        if os.path.isdir(fig_dir):
            figs = sorted(glob.glob(os.path.join(fig_dir, "*.png")))
            if figs:
                st.divider()
                st.subheader("🖼️ Generated Analysis Figures")
                for fig_path in figs:
                    fig_img = Image.open(fig_path)
                    st.image(fig_img, caption=os.path.basename(fig_path), use_container_width=True)

else:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        with st.spinner("🔍 Analyzing video frames... This may take a moment."):
            try:
                detector = load_detector()
                result = detector.analyze_video(tmp_path, generate_report=False)
            finally:
                os.unlink(tmp_path)

        scores = result["scores"]
        verdict = result["verdict"]

        verdict_class = "fake" if verdict == "FAKE" else "real"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="score-big verdict-{verdict_class}">{verdict}</div>'
            f'<p style="font-size:1.3em;">{scores["overall_score"]:.1%} forgery probability</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        cols = st.columns(4)
        labels = ["🎯 Overall", "🧠 Neural Net", "📡 Frequency", "⏱️ Temporal"]
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

        radar_scores = {k.replace("_score", "").replace("_", " ").title(): v for k, v in scores.items() if k != "overall_score"}
        st.plotly_chart(render_radar_chart(radar_scores), use_container_width=True)

st.divider()
st.markdown(
    '<p style="text-align:center; color:#888;">DeepFake Forensics Analyzer v1.0.0 — '
    'Built with PyTorch, OpenCV, and Streamlit</p>',
    unsafe_allow_html=True,
)
