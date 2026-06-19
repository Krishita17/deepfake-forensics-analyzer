import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.stats import kurtosis, skew


class GANArtifactDetector:
    """Detects GAN-specific artifacts: spectral peaks, checkerboard patterns,
    color histogram anomalies, and pixel-level statistical irregularities."""

    def detect_spectral_peaks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float64)

        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        center_mask = np.ones_like(magnitude, dtype=bool)
        Y, X = np.ogrid[:h, :w]
        center_dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        center_mask[center_dist < 5] = False

        masked_mag = magnitude.copy()
        masked_mag[~center_mask] = 0

        mean_val = masked_mag[center_mask].mean()
        std_val = masked_mag[center_mask].std()
        threshold = mean_val + 4.0 * std_val
        peak_count = np.sum(masked_mag > threshold)

        peak_ratio = peak_count / center_mask.sum()

        corners = [
            magnitude[:h//4, :w//4],
            magnitude[:h//4, 3*w//4:],
            magnitude[3*h//4:, :w//4],
            magnitude[3*h//4:, 3*w//4:],
        ]
        corner_energy = np.mean([c.mean() for c in corners])
        center_energy = magnitude[h//4:3*h//4, w//4:3*w//4].mean()
        corner_ratio = corner_energy / (center_energy + 1e-10)

        mid_h = magnitude[cy, :]
        mid_v = magnitude[:, cx]
        cross_energy = (mid_h.mean() + mid_v.mean()) / 2
        non_cross = magnitude.mean()
        cross_ratio = cross_energy / (non_cross + 1e-10)

        return {
            "spectral_peak_count": int(peak_count),
            "spectral_peak_ratio": float(peak_ratio),
            "corner_energy_ratio": float(corner_ratio),
            "cross_pattern_ratio": float(cross_ratio),
            "spectral_score": float(np.clip(
                0.3 * min(peak_ratio * 5000, 1.0)
                + 0.3 * min(corner_ratio * 2, 1.0)
                + 0.4 * min(max(cross_ratio - 1.0, 0) * 2, 1.0),
                0, 1,
            )),
        }

    def detect_checkerboard(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float64)

        cb_kernel = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        response = cv2.filter2D(gray, cv2.CV_64F, cb_kernel)
        cb_energy = np.mean(response ** 2)

        smooth_kernel = np.ones((3, 3), dtype=np.float64) / 9
        smooth_response = cv2.filter2D(gray, cv2.CV_64F, smooth_kernel)
        smooth_energy = np.mean((gray - smooth_response) ** 2)

        ratio = cb_energy / (smooth_energy + 1e-10)

        even_pixels = gray[::2, ::2]
        odd_pixels = gray[1::2, 1::2]
        pattern_diff = abs(even_pixels.mean() - odd_pixels.mean())

        return {
            "checkerboard_energy": float(cb_energy),
            "checkerboard_ratio": float(ratio),
            "pixel_pattern_diff": float(pattern_diff),
            "checkerboard_score": float(np.clip(
                0.5 * min(ratio / 5, 1.0)
                + 0.5 * min(pattern_diff / 5, 1.0),
                0, 1,
            )),
        }

    def analyze_color_statistics(self, image):
        if len(image.shape) != 3:
            return {"color_score": 0.5}

        channels = cv2.split(image)
        channel_stats = {}

        for i, (name, ch) in enumerate(zip(["blue", "green", "red"], channels)):
            ch_float = ch.astype(np.float64).flatten()
            channel_stats[name] = {
                "mean": float(ch_float.mean()),
                "std": float(ch_float.std()),
                "skewness": float(skew(ch_float)),
                "kurtosis": float(kurtosis(ch_float)),
            }

        kurt_values = [channel_stats[c]["kurtosis"] for c in ["blue", "green", "red"]]
        skew_values = [channel_stats[c]["skewness"] for c in ["blue", "green", "red"]]

        kurt_uniformity = np.std(kurt_values) / (np.mean(np.abs(kurt_values)) + 1e-10)
        skew_uniformity = np.std(skew_values) / (np.mean(np.abs(skew_values)) + 1e-10)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float64)
        sat_kurtosis = float(kurtosis(saturation.flatten()))

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        a_channel = lab[:, :, 1].astype(np.float64).flatten()
        b_channel = lab[:, :, 2].astype(np.float64).flatten()
        ab_correlation = float(np.corrcoef(a_channel, b_channel)[0, 1])

        color_score = np.clip(
            0.25 * min(kurt_uniformity, 1.0)
            + 0.25 * min(abs(sat_kurtosis) / 5, 1.0)
            + 0.25 * min(skew_uniformity, 1.0)
            + 0.25 * (1.0 - min(abs(ab_correlation), 1.0)),
            0, 1,
        )

        return {
            "channel_stats": channel_stats,
            "kurtosis_uniformity": float(kurt_uniformity),
            "skewness_uniformity": float(skew_uniformity),
            "saturation_kurtosis": float(sat_kurtosis),
            "ab_correlation": float(ab_correlation),
            "color_score": float(color_score),
        }

    def analyze_noise_patterns(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float64)

        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_residual = gray - denoised

        noise_std = noise_residual.std()
        noise_kurtosis = float(kurtosis(noise_residual.flatten()))
        noise_skewness = float(skew(noise_residual.flatten()))

        h, w = noise_residual.shape
        block_size = 32
        block_stds = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = noise_residual[y:y+block_size, x:x+block_size]
                block_stds.append(block.std())

        block_stds = np.array(block_stds)
        noise_uniformity = block_stds.std() / (block_stds.mean() + 1e-10) if len(block_stds) > 1 else 0

        local_std = cv2.blur(noise_residual ** 2, (16, 16)) ** 0.5
        noise_variance_map = local_std

        auto_corr_h = np.mean(noise_residual[:, :-1] * noise_residual[:, 1:])
        auto_corr_v = np.mean(noise_residual[:-1, :] * noise_residual[1:, :])
        spatial_correlation = (abs(auto_corr_h) + abs(auto_corr_v)) / 2

        noise_score = np.clip(
            0.25 * (1.0 - min(noise_uniformity, 1.0))
            + 0.25 * min(abs(noise_kurtosis) / 10, 1.0)
            + 0.25 * min(spatial_correlation / 2, 1.0)
            + 0.25 * min(abs(noise_skewness) / 2, 1.0),
            0, 1,
        )

        return {
            "noise_std": float(noise_std),
            "noise_kurtosis": float(noise_kurtosis),
            "noise_skewness": float(noise_skewness),
            "noise_uniformity": float(noise_uniformity),
            "spatial_correlation": float(spatial_correlation),
            "noise_score": float(noise_score),
        }

    def detect_upsampling_artifacts(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = gray.astype(np.float64)

        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude = np.abs(f_shift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        periodic_scores = []
        for period in [2, 4, 8]:
            if cy + h // period < h and cx + w // period < w:
                peak_val = magnitude[cy + h // period, cx]
                peak_val += magnitude[cy, cx + w // period]
                peak_val += magnitude[cy - h // period, cx]
                peak_val += magnitude[cy, cx - w // period]
                local_mean = magnitude[
                    max(0, cy + h // period - 3):cy + h // period + 3,
                    max(0, cx - 3):cx + 3
                ].mean()
                periodic_scores.append(peak_val / (4 * local_mean + 1e-10))

        upsample_indicator = max(periodic_scores) if periodic_scores else 0

        return {
            "periodic_artifact_scores": [float(s) for s in periodic_scores],
            "upsampling_score": float(np.clip(min(max(upsample_indicator - 1, 0), 1.0), 0, 1)),
        }

    def compute_stylegan_indicators(self, checkerboard, noise, color):
        cb_energy = checkerboard["checkerboard_energy"]
        low_cb = 1.0 - np.clip(cb_energy / 200.0, 0, 1)

        spatial_corr = noise["spatial_correlation"]
        low_corr = 1.0 - np.clip(spatial_corr / 15.0, 0, 1)

        sat_kurt = color["saturation_kurtosis"]
        high_sat_kurt = np.clip(sat_kurt / 2.0, 0, 1)

        noise_std = noise["noise_std"]
        low_noise = 1.0 - np.clip(noise_std / 10.0, 0, 1)

        return {
            "low_checkerboard_energy": float(low_cb),
            "low_spatial_correlation": float(low_corr),
            "high_saturation_kurtosis": float(high_sat_kurt),
            "low_noise_std": float(low_noise),
            "stylegan_score": float(np.clip(
                0.30 * low_cb
                + 0.30 * low_corr
                + 0.25 * high_sat_kurt
                + 0.15 * low_noise,
                0, 1,
            )),
        }

    def analyze(self, image):
        spectral = self.detect_spectral_peaks(image)
        checkerboard = self.detect_checkerboard(image)
        color = self.analyze_color_statistics(image)
        noise = self.analyze_noise_patterns(image)
        upsample = self.detect_upsampling_artifacts(image)
        stylegan = self.compute_stylegan_indicators(checkerboard, noise, color)

        gan_score = (
            0.15 * spectral["spectral_score"]
            + 0.10 * checkerboard["checkerboard_score"]
            + 0.15 * color["color_score"]
            + 0.15 * noise["noise_score"]
            + 0.10 * upsample["upsampling_score"]
            + 0.35 * stylegan["stylegan_score"]
        )

        return {
            "spectral": spectral,
            "checkerboard": checkerboard,
            "color": color,
            "noise": noise,
            "upsampling": upsample,
            "stylegan": stylegan,
            "gan_artifact_score": float(np.clip(gan_score, 0, 1)),
        }
