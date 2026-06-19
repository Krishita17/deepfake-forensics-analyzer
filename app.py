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
    .compare-card {
        border-radius: 12px; padding: 16px;
        border: 2px solid #2D2D44; text-align: center;
        margin: 8px 0;
    }
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
        title={"text": title, "font": {"size": 16}},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#FF4757" if score > 0.5 else "#00D4AA"},
            "steps": [
                {"range": [0, 30], "color": "#e8f5e9"},
                {"range": [30, 70], "color": "#fff3e0"},
                {"range": [70, 100], "color": "#ffebee"},
            ],
            "threshold": {"line": {"color": "#FFA502", "width": 3}, "thickness": 0.8, "value": 50},
        },
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
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
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=350, margin=dict(t=30, b=30, l=80, r=80), showlegend=False,
    )
    return fig


def render_bar_chart(scores):
    import plotly.graph_objects as go
    names = [k.replace("_score", "").replace("_", " ").title() for k in scores.keys()]
    values = list(scores.values())
    colors = ["#FF4757" if v > 0.5 else "#00D4AA" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors, text=[f"{v:.1%}" for v in values], textposition="outside",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#FFA502", line_width=2)
    fig.update_layout(xaxis=dict(range=[0, 1], title="Score"), height=280, margin=dict(t=10, b=40, l=140, r=40))
    return fig


def render_timeline(frame_scores):
    import plotly.graph_objects as go
    colors = ["#FF4757" if s > 0.5 else "#00D4AA" for s in frame_scores]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=frame_scores, mode="lines+markers",
        line=dict(color="#5B8DEF", width=2), marker=dict(color=colors, size=6),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#FFA502", annotation_text="Threshold")
    fig.update_layout(
        xaxis=dict(title="Frame"), yaxis=dict(title="Fake Probability", range=[0, 1]),
        height=300, margin=dict(t=20, b=50, l=50, r=30),
    )
    return fig


def get_sample_images(n=8):
    samples = {"fake": [], "real": []}
    for label in ["fake", "real"]:
        folder = f"data/processed/test/{label}"
        if os.path.isdir(folder):
            files = sorted(glob.glob(os.path.join(folder, "*")))[:n]
            samples[label] = files
    return samples


def analyze_single(image_path):
    detector = load_detector()
    return detector.analyze_image(image_path, generate_report=True)


def show_frequency_visuals(freq):
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("**FFT Magnitude Spectrum**")
        mag = freq["fft"]["magnitude_spectrum"]
        mag_norm = ((mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255).astype(np.uint8)
        mag_color = cv2.applyColorMap(mag_norm, cv2.COLORMAP_INFERNO)
        st.image(cv2.cvtColor(mag_color, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_f2:
        st.markdown("**FFT Phase Spectrum**")
        phase = freq["fft"]["phase_spectrum"]
        phase_norm = ((phase - phase.min()) / (phase.max() - phase.min() + 1e-8) * 255).astype(np.uint8)
        phase_color = cv2.applyColorMap(phase_norm, cv2.COLORMAP_TWILIGHT)
        st.image(cv2.cvtColor(phase_color, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_f3:
        st.markdown("**DCT Energy Map**")
        dct = freq["dct"]["dct_energy_map"]
        dct_img = np.log1p(dct)
        dct_norm = ((dct_img - dct_img.min()) / (dct_img.max() - dct_img.min() + 1e-8) * 255).astype(np.uint8)
        dct_resized = cv2.resize(dct_norm, (256, 256), interpolation=cv2.INTER_NEAREST)
        dct_color = cv2.applyColorMap(dct_resized, cv2.COLORMAP_MAGMA)
        st.image(cv2.cvtColor(dct_color, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("**Anomaly Indicators**")
    for name, val in freq["anomaly_indicators"].items():
        label = name.replace("_", " ").title()
        col_l, col_b = st.columns([1, 3])
        with col_l:
            st.write(f"**{label}**")
        with col_b:
            st.progress(min(float(val), 1.0), text=f"{val:.3f}")


def show_gan_analysis(gan):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔍 Spectral Analysis**")
        st.metric("Peak Count", gan["spectral"]["spectral_peak_count"])
        st.metric("Corner Energy Ratio", f"{gan['spectral']['corner_energy_ratio']:.3f}")
        st.metric("Spectral Score", f"{gan['spectral']['spectral_score']:.3f}")
    with col2:
        st.markdown("**🔊 Noise Analysis**")
        st.metric("Noise Std", f"{gan['noise']['noise_std']:.3f}")
        st.metric("Noise Uniformity", f"{gan['noise']['noise_uniformity']:.3f}")
        st.metric("Spatial Correlation", f"{gan['noise']['spatial_correlation']:.4f}")
    with col3:
        st.markdown("**🎨 Color Statistics**")
        st.metric("Kurtosis Uniformity", f"{gan['color']['kurtosis_uniformity']:.3f}")
        st.metric("Saturation Kurtosis", f"{gan['color']['saturation_kurtosis']:.3f}")
        st.metric("A-B Correlation", f"{gan['color']['ab_correlation']:.3f}")

    col_cb, col_up = st.columns(2)
    with col_cb:
        st.markdown("**♟️ Checkerboard Detection**")
        st.metric("Checkerboard Score", f"{gan['checkerboard']['checkerboard_score']:.3f}")
    with col_up:
        st.markdown("**⬆️ Upsampling Artifacts**")
        st.metric("Upsampling Score", f"{gan['upsampling']['upsampling_score']:.3f}")


def show_facial_analysis(facial):
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


# ─── SIDEBAR ───
with st.sidebar:
    st.header("⚙️ Settings")
    page = st.selectbox("Mode", ["🔍 Single Analysis", "⚖️ Compare Real vs Fake", "📦 Batch Analysis"])
    analysis_type = st.selectbox("Input Type", ["Image", "Video"])
    show_frequency = st.checkbox("Show Frequency Analysis", True)
    show_gan = st.checkbox("Show GAN Artifact Analysis", True)
    show_facial = st.checkbox("Show Facial Forensics", True)
    show_figures = st.checkbox("Show Generated Figures", True)

    st.divider()
    st.markdown("""
    **Analysis Pipeline:**
    1. 🧠 EfficientNet-B4 + SRM
    2. 📡 FFT / DCT / Wavelet
    3. 🔍 GAN Artifact Detection
    4. 👤 Facial Forensics
    5. ⏱️ Temporal Analysis (Video)
    """)


st.title("🔬 DeepFake Forensics Analyzer")

# ─── SINGLE ANALYSIS ───
if page == "🔍 Single Analysis":
    st.markdown("*Upload or pick a sample image/video for full forensic analysis.*")

    if analysis_type == "Image":
        input_method = st.radio("Image source:", ["Upload a file", "Use sample from dataset"], horizontal=True)
        image_path = None
        is_temp = False

        if input_method == "Upload a file":
            uploaded = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if uploaded is not None:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(uploaded.getvalue())
                tmp.close()
                image_path = tmp.name
                is_temp = True
        else:
            samples = get_sample_images()
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("**🟢 Real samples**")
                for p in samples["real"]:
                    if st.button(os.path.basename(p), key=f"r_{p}"):
                        st.session_state["sel"] = p
            with col_s2:
                st.markdown("**🔴 Fake samples**")
                for p in samples["fake"]:
                    if st.button(os.path.basename(p), key=f"f_{p}"):
                        st.session_state["sel"] = p
            if "sel" in st.session_state:
                image_path = st.session_state["sel"]

        if image_path:
            st.divider()
            image = Image.open(image_path)
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Input Image")
                st.image(image, use_container_width=True)

            with col2:
                st.subheader("Verdict")
                with st.spinner("🔍 Running full forensic analysis..."):
                    result = analyze_single(image_path)
                if is_temp:
                    os.unlink(image_path)

                scores = result["scores"]
                verdict = result["verdict"]
                vc = "fake" if verdict == "FAKE" else "real"
                st.markdown(
                    f'<div class="metric-card"><div class="score-big verdict-{vc}">{verdict}</div>'
                    f'<p style="font-size:1.3em;">{scores["overall_score"]:.1%} forgery probability</p>'
                    f'<p>Confidence: {result["confidence"]:.1%}</p></div>',
                    unsafe_allow_html=True,
                )

            st.divider()
            st.subheader("📊 Score Breakdown")

            col_a, col_b, col_c, col_d, col_e = st.columns(5)
            with col_a:
                st.metric("🎯 Overall", f"{scores['overall_score']:.1%}")
            with col_b:
                st.metric("🧠 Neural", f"{scores['neural_score']:.1%}")
            with col_c:
                st.metric("📡 Frequency", f"{scores['frequency_score']:.1%}")
            with col_d:
                st.metric("👤 Facial", f"{scores['facial_score']:.1%}")
            with col_e:
                st.metric("🔍 GAN", f"{scores['gan_artifact_score']:.1%}")

            st.divider()
            st.subheader("📈 Visual Analysis")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.plotly_chart(render_gauge_chart(scores["overall_score"], "Overall Forgery Score"), use_container_width=True)
            with col_g2:
                radar = {k.replace("_score", "").replace("_", " ").title(): v for k, v in scores.items() if k != "overall_score"}
                st.plotly_chart(render_radar_chart(radar), use_container_width=True)

            st.plotly_chart(render_bar_chart({k: v for k, v in scores.items() if k != "overall_score"}), use_container_width=True)

            if show_gan and "gan_analysis" in result:
                with st.expander("🔍 GAN Artifact Analysis", expanded=True):
                    show_gan_analysis(result["gan_analysis"])

            if show_frequency and "frequency_analysis" in result:
                with st.expander("📡 Frequency Domain Analysis", expanded=True):
                    show_frequency_visuals(result["frequency_analysis"])

            if show_facial and "facial_analysis" in result:
                with st.expander("👤 Facial Forensics", expanded=False):
                    show_facial_analysis(result["facial_analysis"])

            if show_figures:
                fig_dir = "outputs/figures"
                if os.path.isdir(fig_dir):
                    figs = sorted(glob.glob(os.path.join(fig_dir, "*.png")))
                    if figs:
                        with st.expander("🖼️ Generated Analysis Figures", expanded=False):
                            for fig_path in figs:
                                st.image(Image.open(fig_path), caption=os.path.basename(fig_path), use_container_width=True)

    else:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            st.video(tmp_path)
            with st.spinner("🔍 Analyzing video frames..."):
                detector = load_detector()
                result = detector.analyze_video(tmp_path, generate_report=False)
                os.unlink(tmp_path)

            scores = result["scores"]
            verdict = result["verdict"]
            vc = "fake" if verdict == "FAKE" else "real"
            st.markdown(
                f'<div class="metric-card"><div class="score-big verdict-{vc}">{verdict}</div>'
                f'<p>{scores["overall_score"]:.1%} forgery probability</p></div>',
                unsafe_allow_html=True,
            )
            st.divider()
            cols = st.columns(4)
            for col, (label, key) in zip(cols, [("Overall", "overall_score"), ("Neural", "neural_score"), ("Frequency", "frequency_score"), ("Temporal", "temporal_score")]):
                with col:
                    st.metric(label, f"{scores.get(key, 0):.1%}")

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(render_gauge_chart(scores["overall_score"]), use_container_width=True)
            with col_v2:
                if result.get("frame_scores"):
                    st.plotly_chart(render_timeline(result["frame_scores"]), use_container_width=True)

# ─── COMPARE MODE ───
elif page == "⚖️ Compare Real vs Fake":
    st.markdown("*Side-by-side comparison of a real and fake image.*")

    samples = get_sample_images(3)
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🟢 Real Image")
        real_choice = st.selectbox("Pick a real sample:", samples["real"], format_func=os.path.basename)

    with col_right:
        st.subheader("🔴 Fake Image")
        fake_choice = st.selectbox("Pick a fake sample:", samples["fake"], format_func=os.path.basename)

    if st.button("⚡ Analyze Both", use_container_width=True):
        col_r, col_f = st.columns(2)
        with col_r:
            st.image(Image.open(real_choice), use_container_width=True)
            with st.spinner("Analyzing real..."):
                result_real = analyze_single(real_choice)
            sr = result_real["scores"]
            vr = result_real["verdict"]
            st.markdown(
                f'<div class="compare-card" style="border-color:{"#00D4AA" if vr == "REAL" else "#FF4757"}">'
                f'<h2 style="color:{"#00D4AA" if vr == "REAL" else "#FF4757"}">{vr}</h2>'
                f'<p>Overall: {sr["overall_score"]:.1%}</p></div>',
                unsafe_allow_html=True,
            )
            for k, v in sr.items():
                if k != "overall_score":
                    st.progress(min(v, 1.0), text=f"{k.replace('_score','').replace('_',' ').title()}: {v:.1%}")

        with col_f:
            st.image(Image.open(fake_choice), use_container_width=True)
            with st.spinner("Analyzing fake..."):
                result_fake = analyze_single(fake_choice)
            sf = result_fake["scores"]
            vf = result_fake["verdict"]
            st.markdown(
                f'<div class="compare-card" style="border-color:{"#00D4AA" if vf == "REAL" else "#FF4757"}">'
                f'<h2 style="color:{"#00D4AA" if vf == "REAL" else "#FF4757"}">{vf}</h2>'
                f'<p>Overall: {sf["overall_score"]:.1%}</p></div>',
                unsafe_allow_html=True,
            )
            for k, v in sf.items():
                if k != "overall_score":
                    st.progress(min(v, 1.0), text=f"{k.replace('_score','').replace('_',' ').title()}: {v:.1%}")

        st.divider()
        st.subheader("📊 Score Comparison")
        import plotly.graph_objects as go
        categories = [k.replace("_score", "").replace("_", " ").title() for k in sr.keys()]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Real Image", x=categories, y=list(sr.values()), marker_color="#00D4AA"))
        fig.add_trace(go.Bar(name="Fake Image", x=categories, y=list(sf.values()), marker_color="#FF4757"))
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 1], title="Score"), height=400)
        st.plotly_chart(fig, use_container_width=True)

# ─── BATCH ANALYSIS ───
elif page == "📦 Batch Analysis":
    st.markdown("*Analyze multiple images at once and see summary statistics.*")

    uploaded_files = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    use_samples = st.checkbox("Or use test dataset samples instead")
    n_samples = st.slider("Number of samples (per class)", 1, 10, 3) if use_samples else 0

    if st.button("🚀 Run Batch Analysis", use_container_width=True):
        paths = []
        labels = []
        temp_paths = []

        if use_samples:
            samples = get_sample_images(n_samples)
            for p in samples["real"]:
                paths.append(p)
                labels.append("real")
            for p in samples["fake"]:
                paths.append(p)
                labels.append("fake")
        elif uploaded_files:
            for uf in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(uf.getvalue())
                tmp.close()
                paths.append(tmp.name)
                labels.append("unknown")
                temp_paths.append(tmp.name)

        if paths:
            results = []
            progress = st.progress(0, text="Analyzing...")
            for i, (path, label) in enumerate(zip(paths, labels)):
                progress.progress((i + 1) / len(paths), text=f"Analyzing {i+1}/{len(paths)}...")
                r = analyze_single(path)
                r["source_label"] = label
                r["filename"] = os.path.basename(path)
                results.append(r)

            for tp in temp_paths:
                os.unlink(tp)

            st.divider()
            st.subheader("📊 Results Summary")

            for r in results:
                s = r["scores"]
                v = r["verdict"]
                icon = "✅" if (r["source_label"] == "real" and v == "REAL") or (r["source_label"] == "fake" and v == "FAKE") else "❌" if r["source_label"] != "unknown" else "📄"
                color = "#00D4AA" if v == "REAL" else "#FF4757"

                with st.container():
                    c1, c2, c3, c4, c5, c6 = st.columns([1, 2, 1, 1, 1, 1])
                    with c1:
                        st.write(f"{icon}")
                    with c2:
                        st.write(f"**{r['filename']}**")
                    with c3:
                        st.markdown(f"<span style='color:{color};font-weight:bold'>{v}</span>", unsafe_allow_html=True)
                    with c4:
                        st.write(f"{s['overall_score']:.1%}")
                    with c5:
                        st.write(f"GAN: {s['gan_artifact_score']:.1%}")
                    with c6:
                        st.write(f"Label: {r['source_label']}")

            if any(r["source_label"] != "unknown" for r in results):
                correct = sum(
                    1 for r in results
                    if (r["source_label"] == "real" and r["verdict"] == "REAL")
                    or (r["source_label"] == "fake" and r["verdict"] == "FAKE")
                )
                total = sum(1 for r in results if r["source_label"] != "unknown")
                st.divider()
                st.metric("🎯 Accuracy on known labels", f"{correct}/{total} ({correct/total:.0%})")

            st.divider()
            st.subheader("📈 Score Distribution")
            import plotly.graph_objects as go
            overall_scores = [r["scores"]["overall_score"] for r in results]
            filenames = [r["filename"] for r in results]
            colors = ["#FF4757" if s > 0.5 else "#00D4AA" for s in overall_scores]
            fig = go.Figure(go.Bar(x=filenames, y=overall_scores, marker_color=colors))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#FFA502")
            fig.update_layout(yaxis=dict(range=[0, 1], title="Forgery Score"), xaxis=dict(title="Image"), height=400)
            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown(
    '<p style="text-align:center; color:#888;">DeepFake Forensics Analyzer v1.0.0</p>',
    unsafe_allow_html=True,
)
