import argparse
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path

from src.models import EnsembleDetector, EfficientNetDetector
from src.preprocessing import FaceExtractor, get_validation_transforms
from src.analysis import FrequencyAnalyzer, FacialForensicsAnalyzer, TemporalConsistencyAnalyzer, GradCAMExplainer
from src.visualization import ForensicsPlotter, ReportGenerator


class DeepfakeDetector:
    def __init__(self, config_path="configs/default.yaml", weights_path="weights/best_model.pth"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_validation_transforms(self.config["data"]["image_size"])
        self.face_extractor = FaceExtractor()
        self.freq_analyzer = FrequencyAnalyzer(
            dct_block_size=self.config["analysis"]["frequency"]["dct_block_size"],
            wavelet=self.config["analysis"]["frequency"]["wavelet"],
            wavelet_levels=self.config["analysis"]["frequency"]["wavelet_levels"],
        )
        self.facial_analyzer = FacialForensicsAnalyzer()
        self.temporal_analyzer = TemporalConsistencyAnalyzer(
            window_size=self.config["analysis"]["temporal"]["window_size"],
            stride=self.config["analysis"]["temporal"]["stride"],
        )
        self.plotter = ForensicsPlotter()
        self.report_gen = ReportGenerator()

        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path):
        if self.config["model"]["ensemble"]["enabled"]:
            model = EnsembleDetector(self.config)
        else:
            model = EfficientNetDetector(
                model_name=self.config["model"]["backbone"],
                num_classes=self.config["model"]["num_classes"],
            )

        weights_file = Path(weights_path)
        if weights_file.exists():
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded weights from {weights_path}")
        else:
            print("No pre-trained weights found. Using random initialization.")

        model = model.to(self.device)
        model.eval()
        return model

    def analyze_image(self, image_path, generate_report=True):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        print(f"Analyzing: {image_path}")

        faces = self.face_extractor.extract_faces(image)
        if not faces:
            print("  No faces detected. Analyzing full image.")
            face_image = cv2.resize(image, (380, 380))
        else:
            face_image = faces[0]["face"]
            print(f"  Detected {len(faces)} face(s), analyzing primary face.")

        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB) if len(face_image.shape) == 3 and face_image.shape[2] == 3 else face_image
        transformed = self.transform(image=rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            model_output = self.model(input_tensor)

        freq_result = self.freq_analyzer.analyze(face_image)
        facial_result = self.facial_analyzer.analyze(face_image)

        neural_score = model_output["probabilities"][0, 1].item()
        freq_score = freq_result["frequency_forgery_score"]
        facial_score = facial_result["facial_forgery_score"]

        overall_score = (
            0.45 * neural_score
            + 0.30 * freq_score
            + 0.25 * facial_score
        )

        scores = {
            "neural_score": neural_score,
            "frequency_score": freq_score,
            "facial_score": facial_score,
            "overall_score": overall_score,
        }

        result = {
            "scores": scores,
            "verdict": "FAKE" if overall_score > 0.5 else "REAL",
            "confidence": abs(overall_score - 0.5) * 2,
            "model_output": model_output,
            "frequency_analysis": freq_result,
            "facial_analysis": facial_result,
            "details": {
                "faces_detected": len(faces),
                "face_confidence": faces[0]["confidence"] if faces else 0,
                "spectral_slope": freq_result["spectral_slope"]["slope"],
                "dct_high_freq_ratio": freq_result["dct"]["high_freq_ratio"],
                "landmark_consistency": facial_result["consistency"].get("consistency_score", 0),
                "blending_artifacts": facial_result["blending"].get("blending_score", 0),
            },
        }

        print(f"\n  Verdict: {result['verdict']} ({overall_score:.1%} forgery probability)")
        print(f"  Neural: {neural_score:.3f} | Frequency: {freq_score:.3f} | Facial: {facial_score:.3f}")

        if generate_report:
            fig_paths = []
            fig_paths.append(self.plotter.plot_frequency_analysis(freq_result))
            fig_paths.append(self.plotter.plot_comprehensive_report(scores))

            self.report_gen.generate_html_report(result, image_paths=fig_paths)
            self.report_gen.generate_json_report(result)
            print("  Reports saved to outputs/reports/")

        return result

    def analyze_video(self, video_path, generate_report=True):
        print(f"Analyzing video: {video_path}")

        faces, video_info = self.face_extractor.extract_from_video(str(video_path), sample_rate=15)
        print(f"  Extracted {len(faces)} face samples from {video_info['total_frames']} frames")

        temporal_result = self.temporal_analyzer.analyze_video(str(video_path))

        frame_scores = []
        for face_data in faces:
            face_img = face_data["face"]
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) if face_img.shape[2] == 3 else face_img
            transformed = self.transform(image=rgb)
            input_tensor = transformed["image"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
            frame_scores.append(output["probabilities"][0, 1].item())

        neural_score = np.mean(frame_scores) if frame_scores else 0.5
        temporal_score = temporal_result["temporal_forgery_score"]
        freq_scores = []
        for face_data in faces[:10]:
            freq_result = self.freq_analyzer.analyze(face_data["face"])
            freq_scores.append(freq_result["frequency_forgery_score"])
        freq_score = np.mean(freq_scores) if freq_scores else 0.5

        overall_score = (
            0.40 * neural_score
            + 0.25 * freq_score
            + 0.35 * temporal_score
        )

        scores = {
            "neural_score": neural_score,
            "frequency_score": freq_score,
            "temporal_score": temporal_score,
            "overall_score": overall_score,
        }

        result = {
            "scores": scores,
            "verdict": "FAKE" if overall_score > 0.5 else "REAL",
            "confidence": abs(overall_score - 0.5) * 2,
            "frame_scores": frame_scores,
            "temporal_analysis": temporal_result,
            "video_info": video_info,
        }

        print(f"\n  Verdict: {result['verdict']} ({overall_score:.1%} forgery probability)")

        if generate_report:
            fig_paths = []
            fig_paths.append(self.plotter.plot_temporal_analysis(temporal_result))
            fig_paths.append(self.plotter.plot_comprehensive_report(scores))

            self.report_gen.generate_html_report(result, image_paths=fig_paths)
            self.report_gen.generate_json_report(result)

        return result


def main():
    parser = argparse.ArgumentParser(description="DeepFake Forensics Analyzer")
    parser.add_argument("input", help="Path to image or video file")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--weights", default="weights/best_model.pth", help="Model weights path")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation")
    args = parser.parse_args()

    detector = DeepfakeDetector(config_path=args.config, weights_path=args.weights)

    input_path = Path(args.input)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if input_path.suffix.lower() in video_extensions:
        result = detector.analyze_video(input_path, generate_report=not args.no_report)
    else:
        result = detector.analyze_image(input_path, generate_report=not args.no_report)

    return result


if __name__ == "__main__":
    main()
