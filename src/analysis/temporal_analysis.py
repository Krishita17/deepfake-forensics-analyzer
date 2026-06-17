import numpy as np
import cv2
from collections import deque


class TemporalConsistencyAnalyzer:
    def __init__(self, window_size=16, stride=4):
        self.window_size = window_size
        self.stride = stride

    def compute_optical_flow(self, prev_frame, curr_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return flow, magnitude, angle

    def analyze_flow_consistency(self, magnitudes, angles):
        if len(magnitudes) < 2:
            return {"temporal_smoothness": 1.0, "flow_variance": 0.0}

        mag_diffs = []
        angle_diffs = []
        for i in range(1, len(magnitudes)):
            mag_diffs.append(np.mean(np.abs(magnitudes[i] - magnitudes[i - 1])))
            angle_diff = np.abs(angles[i] - angles[i - 1])
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            angle_diffs.append(np.mean(angle_diff))

        mag_variance = np.var(mag_diffs)
        angle_variance = np.var(angle_diffs)

        smoothness = 1.0 / (1.0 + mag_variance + angle_variance)

        return {
            "temporal_smoothness": float(smoothness),
            "magnitude_variance": float(mag_variance),
            "angle_variance": float(angle_variance),
            "flow_variance": float(mag_variance + angle_variance),
            "magnitude_diffs": [float(d) for d in mag_diffs],
        }

    def detect_flickering(self, frames, region_mask=None):
        intensities = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if region_mask is not None:
                gray = cv2.bitwise_and(gray, gray, mask=region_mask)
            intensities.append(gray.mean())

        intensities = np.array(intensities)
        if len(intensities) < 3:
            return {"flicker_score": 0.0}

        diffs = np.abs(np.diff(intensities))
        second_diffs = np.abs(np.diff(diffs))

        flicker_score = np.mean(second_diffs) / (np.mean(intensities) + 1e-10)

        from scipy.signal import welch
        if len(intensities) >= 8:
            freqs, psd = welch(intensities, fs=30.0, nperseg=min(len(intensities), 256))
            high_freq_mask = freqs > 5.0
            high_freq_energy = psd[high_freq_mask].sum() if high_freq_mask.any() else 0
            total_energy = psd.sum() + 1e-10
            freq_ratio = high_freq_energy / total_energy
        else:
            freq_ratio = 0.0

        return {
            "flicker_score": float(np.clip(flicker_score * 10, 0, 1)),
            "intensity_variance": float(np.var(intensities)),
            "high_freq_ratio": float(freq_ratio),
            "intensity_timeline": intensities.tolist(),
        }

    def analyze_face_tracking_consistency(self, face_locations):
        if len(face_locations) < 3:
            return {"tracking_score": 1.0}

        centers = np.array(face_locations)
        velocities = np.diff(centers, axis=0)
        accelerations = np.diff(velocities, axis=0)

        velocity_smoothness = 1.0 / (1.0 + np.mean(np.abs(accelerations)))

        speed = np.linalg.norm(velocities, axis=1)
        speed_variance = np.var(speed) / (np.mean(speed) + 1e-10)

        direction = np.arctan2(velocities[:, 1], velocities[:, 0])
        direction_changes = np.abs(np.diff(direction))
        direction_smoothness = 1.0 / (1.0 + np.mean(direction_changes))

        return {
            "tracking_score": float(np.clip(
                0.4 * velocity_smoothness + 0.3 * (1.0 / (1.0 + speed_variance)) + 0.3 * direction_smoothness,
                0, 1,
            )),
            "velocity_smoothness": float(velocity_smoothness),
            "speed_variance": float(speed_variance),
            "direction_smoothness": float(direction_smoothness),
        }

    def analyze_video(self, video_path, face_detector=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = deque(maxlen=self.window_size)
        magnitudes = []
        angles = []
        face_locations = []
        all_flicker_scores = []

        frame_idx = 0
        prev_frame = None
        results_per_window = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (320, 240))
            frames.append(frame)

            if prev_frame is not None:
                _, mag, ang = self.compute_optical_flow(prev_frame, frame)
                magnitudes.append(mag)
                angles.append(ang)

            if face_detector is not None:
                face_loc = face_detector(frame)
                if face_loc is not None:
                    face_locations.append(face_loc)

            if len(frames) == self.window_size and frame_idx % self.stride == 0:
                window_frames = list(frames)
                flicker = self.detect_flickering(window_frames)
                all_flicker_scores.append(flicker["flicker_score"])

                if len(magnitudes) >= 2:
                    window_mags = magnitudes[-self.window_size:]
                    window_angs = angles[-self.window_size:]
                    flow_result = self.analyze_flow_consistency(window_mags, window_angs)
                    results_per_window.append({
                        "frame": frame_idx,
                        "flow": flow_result,
                        "flicker": flicker,
                    })

            prev_frame = frame
            frame_idx += 1

        cap.release()

        tracking = self.analyze_face_tracking_consistency(face_locations) if face_locations else {"tracking_score": 1.0}

        avg_smoothness = np.mean([r["flow"]["temporal_smoothness"] for r in results_per_window]) if results_per_window else 1.0
        avg_flicker = np.mean(all_flicker_scores) if all_flicker_scores else 0.0

        temporal_forgery_score = (
            0.35 * (1.0 - avg_smoothness)
            + 0.30 * avg_flicker
            + 0.35 * (1.0 - tracking["tracking_score"])
        )

        return {
            "fps": fps,
            "total_frames": total_frames,
            "windows_analyzed": len(results_per_window),
            "temporal_smoothness": float(avg_smoothness),
            "flicker_score": float(avg_flicker),
            "face_tracking": tracking,
            "per_window_results": results_per_window,
            "temporal_forgery_score": float(np.clip(temporal_forgery_score, 0, 1)),
        }
