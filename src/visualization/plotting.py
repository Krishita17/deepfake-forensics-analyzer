import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter
import os


class ForensicsPlotter:
    def __init__(self, output_dir="outputs/figures", style="dark_background"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use(style)
        self.colors = {
            "real": "#00D4AA",
            "fake": "#FF4757",
            "neutral": "#5B8DEF",
            "warning": "#FFA502",
            "bg": "#1A1A2E",
            "text": "#E8E8E8",
            "grid": "#2D2D44",
        }

    def plot_frequency_analysis(self, analysis_result, save_name="frequency_analysis.png"):
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor(self.colors["bg"])
        gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0:2])
        mag = analysis_result["fft"]["magnitude_spectrum"]
        im = ax1.imshow(mag, cmap="inferno", aspect="auto")
        ax1.set_title("FFT Magnitude Spectrum", color=self.colors["text"], fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(gs[0, 2:4])
        phase = analysis_result["fft"]["phase_spectrum"]
        im2 = ax2.imshow(phase, cmap="twilight", aspect="auto")
        ax2.set_title("FFT Phase Spectrum", color=self.colors["text"], fontsize=13, fontweight="bold")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        ax3 = fig.add_subplot(gs[1, 0:2])
        radial = analysis_result["radial_profile"]
        freqs = np.arange(len(radial))
        ax3.fill_between(freqs, radial, alpha=0.3, color=self.colors["neutral"])
        ax3.plot(freqs, radial, color=self.colors["neutral"], linewidth=2)
        slope_info = analysis_result["spectral_slope"]
        fitted = np.exp(slope_info["slope"] * np.log(freqs[1:] + 1) + slope_info["intercept"])
        ax3.plot(freqs[1:], fitted, "--", color=self.colors["warning"], linewidth=2, label=f"Slope: {slope_info['slope']:.2f}")
        ax3.set_xlabel("Spatial Frequency", color=self.colors["text"])
        ax3.set_ylabel("Power", color=self.colors["text"])
        ax3.set_title("Radial Power Spectrum", color=self.colors["text"], fontsize=13, fontweight="bold")
        ax3.legend(facecolor=self.colors["bg"], edgecolor=self.colors["grid"])
        ax3.set_yscale("log")

        ax4 = fig.add_subplot(gs[1, 2:4])
        dct = analysis_result["dct"]["dct_energy_map"]
        im4 = ax4.imshow(np.log1p(dct), cmap="magma", aspect="auto")
        ax4.set_title("DCT Energy Distribution", color=self.colors["text"], fontsize=13, fontweight="bold")
        plt.colorbar(im4, ax=ax4, fraction=0.046)

        ax5 = fig.add_subplot(gs[2, 0:2])
        wavelet = analysis_result["wavelet"]
        levels = list(range(1, len(wavelet["energy_per_level"]) + 1))
        energy = wavelet["energy_per_level"]
        bars = ax5.bar(levels, energy, color=[self.colors["neutral"]] * len(levels), alpha=0.8, edgecolor="white", linewidth=0.5)
        ax5.set_xlabel("Wavelet Level", color=self.colors["text"])
        ax5.set_ylabel("Energy", color=self.colors["text"])
        ax5.set_title("Wavelet Energy per Level", color=self.colors["text"], fontsize=13, fontweight="bold")

        ax6 = fig.add_subplot(gs[2, 2:4])
        indicators = analysis_result["anomaly_indicators"]
        names = list(indicators.keys())
        values = list(indicators.values())
        bar_colors = [self.colors["fake"] if v > 0.5 else self.colors["real"] for v in values]
        bars = ax6.barh(names, values, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax6.axvline(x=0.5, color=self.colors["warning"], linestyle="--", linewidth=1.5, alpha=0.7)
        ax6.set_xlim(0, 1)
        ax6.set_title("Anomaly Indicators", color=self.colors["text"], fontsize=13, fontweight="bold")

        for ax in fig.get_axes():
            ax.tick_params(colors=self.colors["text"])
            ax.set_facecolor(self.colors["bg"])

        fig.suptitle(
            f"Frequency Forensics Analysis — Forgery Score: {analysis_result['frequency_forgery_score']:.2%}",
            color=self.colors["text"], fontsize=16, fontweight="bold", y=0.98,
        )

        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=self.colors["bg"])
        plt.close(fig)
        return path

    def plot_ensemble_results(self, ensemble_output, save_name="ensemble_results.png"):
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor(self.colors["bg"])
        gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        probs = ensemble_output["probabilities"][0].detach().cpu().numpy()
        labels = ["Real", "Fake"]
        colors = [self.colors["real"], self.colors["fake"]]
        wedges, texts, autotexts = ax1.pie(
            probs, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"color": self.colors["text"], "fontsize": 12},
        )
        ax1.set_title("Final Prediction", color=self.colors["text"], fontsize=13, fontweight="bold")

        ax2 = fig.add_subplot(gs[0, 1])
        model_names = []
        model_fake_probs = []
        for name, out in ensemble_output["individual_outputs"].items():
            model_names.append(name.replace("_", "\n"))
            model_fake_probs.append(out["probabilities"][0, 1].item())
        bars = ax2.bar(model_names, model_fake_probs, color=self.colors["neutral"], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, model_fake_probs):
            bar.set_color(self.colors["fake"] if val > 0.5 else self.colors["real"])
        ax2.axhline(y=0.5, color=self.colors["warning"], linestyle="--", linewidth=1.5)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Fake Probability", color=self.colors["text"])
        ax2.set_title("Per-Model Predictions", color=self.colors["text"], fontsize=13, fontweight="bold")

        ax3 = fig.add_subplot(gs[0, 2])
        confidence = ensemble_output["confidence"][0].item()
        agreement = ensemble_output["model_agreement"][0].item()
        metrics = {"Confidence": confidence, "Model\nAgreement": agreement}
        bars = ax3.bar(
            metrics.keys(), metrics.values(),
            color=[self.colors["real"] if v > 0.7 else self.colors["warning"] for v in metrics.values()],
            alpha=0.85, edgecolor="white",
        )
        ax3.set_ylim(0, 1)
        ax3.set_title("Reliability Metrics", color=self.colors["text"], fontsize=13, fontweight="bold")

        ax4 = fig.add_subplot(gs[1, 0:2])
        if isinstance(ensemble_output["uncertainty"], dict):
            unc = ensemble_output["uncertainty"]
            unc_names = list(unc.keys())
            unc_vals = [v[0].item() if hasattr(v, "item") else float(v) for v in unc.values()]
            ax4.barh(unc_names, unc_vals, color=self.colors["warning"], alpha=0.8)
            ax4.set_title("Uncertainty Decomposition", color=self.colors["text"], fontsize=13, fontweight="bold")

        ax5 = fig.add_subplot(gs[1, 2])
        score_text = "FAKE" if probs[1] > 0.5 else "REAL"
        score_color = self.colors["fake"] if probs[1] > 0.5 else self.colors["real"]
        ax5.text(0.5, 0.6, score_text, ha="center", va="center", fontsize=48,
                 fontweight="bold", color=score_color, transform=ax5.transAxes)
        ax5.text(0.5, 0.3, f"{max(probs) * 100:.1f}% confident", ha="center", va="center",
                 fontsize=16, color=self.colors["text"], transform=ax5.transAxes)
        ax5.axis("off")

        for ax in fig.get_axes():
            ax.tick_params(colors=self.colors["text"])
            if ax.get_facecolor() != (0, 0, 0, 0):
                ax.set_facecolor(self.colors["bg"])

        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=self.colors["bg"])
        plt.close(fig)
        return path

    def plot_temporal_analysis(self, temporal_result, save_name="temporal_analysis.png"):
        fig = plt.figure(figsize=(18, 10))
        fig.patch.set_facecolor(self.colors["bg"])
        gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0:2])
        if temporal_result["per_window_results"]:
            frames = [r["frame"] for r in temporal_result["per_window_results"]]
            smoothness = [r["flow"]["temporal_smoothness"] for r in temporal_result["per_window_results"]]
            ax1.plot(frames, smoothness, color=self.colors["neutral"], linewidth=2)
            ax1.fill_between(frames, smoothness, alpha=0.2, color=self.colors["neutral"])
            ax1.axhline(y=0.5, color=self.colors["warning"], linestyle="--")
        ax1.set_xlabel("Frame", color=self.colors["text"])
        ax1.set_ylabel("Temporal Smoothness", color=self.colors["text"])
        ax1.set_title("Optical Flow Consistency Over Time", color=self.colors["text"], fontsize=13, fontweight="bold")

        ax2 = fig.add_subplot(gs[0, 2])
        scores = {
            "Temporal\nSmoothness": temporal_result["temporal_smoothness"],
            "Flicker\nScore": 1.0 - temporal_result["flicker_score"],
            "Face\nTracking": temporal_result["face_tracking"]["tracking_score"],
        }
        theta = np.linspace(0, 2 * np.pi, len(scores) + 1)
        values = list(scores.values()) + [list(scores.values())[0]]
        ax2_polar = fig.add_subplot(gs[0, 2], polar=True)
        ax2_polar.fill(theta, values, alpha=0.3, color=self.colors["neutral"])
        ax2_polar.plot(theta, values, color=self.colors["neutral"], linewidth=2)
        ax2_polar.set_xticks(theta[:-1])
        ax2_polar.set_xticklabels(scores.keys(), color=self.colors["text"], fontsize=9)
        ax2_polar.set_facecolor(self.colors["bg"])
        ax2_polar.set_title("Quality Radar", color=self.colors["text"], fontsize=13, fontweight="bold", pad=20)

        ax3 = fig.add_subplot(gs[1, 0:2])
        if temporal_result["per_window_results"]:
            flicker_scores = [r["flicker"]["flicker_score"] for r in temporal_result["per_window_results"]]
            frames = [r["frame"] for r in temporal_result["per_window_results"]]
            colors = [self.colors["fake"] if s > 0.5 else self.colors["real"] for s in flicker_scores]
            ax3.bar(frames, flicker_scores, color=colors, alpha=0.8, width=max(1, len(frames) // 50))
        ax3.set_xlabel("Frame", color=self.colors["text"])
        ax3.set_ylabel("Flicker Score", color=self.colors["text"])
        ax3.set_title("Flickering Detection Timeline", color=self.colors["text"], fontsize=13, fontweight="bold")

        ax4 = fig.add_subplot(gs[1, 2])
        score = temporal_result["temporal_forgery_score"]
        ax4.text(0.5, 0.55, f"{score:.1%}", ha="center", va="center", fontsize=42,
                 fontweight="bold", color=self.colors["fake"] if score > 0.5 else self.colors["real"],
                 transform=ax4.transAxes)
        ax4.text(0.5, 0.3, "Temporal Forgery Score", ha="center", va="center",
                 fontsize=12, color=self.colors["text"], transform=ax4.transAxes)
        ax4.axis("off")

        for ax in fig.get_axes():
            ax.tick_params(colors=self.colors["text"])

        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=self.colors["bg"])
        plt.close(fig)
        return path

    def plot_comprehensive_report(self, all_scores, save_name="comprehensive_report.png"):
        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor(self.colors["bg"])

        gs = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.3)

        ax_gauge = fig.add_subplot(gs[0, 1:3])
        overall = all_scores.get("overall_score", 0.5)
        theta_range = np.linspace(np.pi, 0, 100)
        ax_gauge.plot(np.cos(theta_range), np.sin(theta_range), color=self.colors["grid"], linewidth=8, solid_capstyle="round")

        gradient = np.linspace(0, 1, 100)
        for i in range(int(overall * 100)):
            color = plt.cm.RdYlGn_r(gradient[i])
            ax_gauge.plot(np.cos(theta_range[i:i+2]), np.sin(theta_range[i:i+2]), color=color, linewidth=8, solid_capstyle="round")

        needle_angle = np.pi * (1 - overall)
        ax_gauge.annotate("", xy=(0.7 * np.cos(needle_angle), 0.7 * np.sin(needle_angle)), xytext=(0, 0),
                          arrowprops=dict(arrowstyle="->", color=self.colors["text"], lw=2))
        ax_gauge.text(0, -0.15, f"{overall:.1%}", ha="center", fontsize=28, fontweight="bold",
                      color=self.colors["fake"] if overall > 0.5 else self.colors["real"])
        ax_gauge.text(0, -0.35, "Deepfake Probability", ha="center", fontsize=12, color=self.colors["text"])
        ax_gauge.set_xlim(-1.2, 1.2)
        ax_gauge.set_ylim(-0.5, 1.2)
        ax_gauge.axis("off")

        ax_bars = fig.add_subplot(gs[1, :])
        score_names = [k.replace("_", " ").title() for k in all_scores.keys() if k != "overall_score"]
        score_values = [v for k, v in all_scores.items() if k != "overall_score"]
        bar_colors = [self.colors["fake"] if v > 0.5 else self.colors["real"] for v in score_values]

        bars = ax_bars.barh(score_names, score_values, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5, height=0.6)
        ax_bars.axvline(x=0.5, color=self.colors["warning"], linestyle="--", linewidth=1.5, alpha=0.7)
        ax_bars.set_xlim(0, 1)

        for bar, val in zip(bars, score_values):
            ax_bars.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.1%}",
                         va="center", color=self.colors["text"], fontsize=11, fontweight="bold")

        ax_bars.set_title("Component Analysis Scores", color=self.colors["text"], fontsize=14, fontweight="bold")
        ax_bars.tick_params(colors=self.colors["text"])
        ax_bars.set_facecolor(self.colors["bg"])

        fig.suptitle("DeepFake Forensics — Comprehensive Analysis Report",
                     color=self.colors["text"], fontsize=16, fontweight="bold", y=0.98)

        path = os.path.join(self.output_dir, save_name)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=self.colors["bg"])
        plt.close(fig)
        return path
