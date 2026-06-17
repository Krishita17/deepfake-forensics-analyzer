import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy.signal import welch
import pywt


class FrequencyAnalyzer:
    def __init__(self, dct_block_size=8, wavelet="db4", wavelet_levels=4):
        self.dct_block_size = dct_block_size
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels

    def compute_fft_spectrum(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        f_transform = fft2(gray.astype(np.float64))
        f_shift = fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))
        phase = np.angle(f_shift)

        return {
            "magnitude_spectrum": magnitude,
            "phase_spectrum": phase,
            "raw_fft": f_shift,
        }

    def compute_azimuthal_average(self, spectrum):
        h, w = spectrum.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)

        max_r = min(cy, cx)
        radial_mean = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            if mask.any():
                radial_mean[i] = spectrum[mask].mean()

        return radial_mean

    def compute_dct_statistics(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        bs = self.dct_block_size
        h, w = gray.shape
        h_blocks = h // bs
        w_blocks = w // bs
        gray = gray[:h_blocks * bs, :w_blocks * bs]

        dct_energy = np.zeros((bs, bs))
        block_count = 0

        for i in range(0, gray.shape[0] - bs + 1, bs):
            for j in range(0, gray.shape[1] - bs + 1, bs):
                block = gray[i:i + bs, j:j + bs].astype(np.float64)
                dct_block = cv2.dct(block)
                dct_energy += np.abs(dct_block)
                block_count += 1

        dct_energy /= max(block_count, 1)

        high_freq_ratio = dct_energy[bs // 2:, bs // 2:].sum() / (dct_energy.sum() + 1e-10)

        return {
            "dct_energy_map": dct_energy,
            "high_freq_ratio": high_freq_ratio,
            "mean_energy": dct_energy.mean(),
            "energy_std": dct_energy.std(),
        }

    def compute_wavelet_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        coeffs = pywt.wavedec2(gray.astype(np.float64), self.wavelet, level=self.wavelet_levels)

        features = {}
        energy_per_level = []

        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            level_energy = {
                "horizontal": np.sqrt(np.mean(cH ** 2)),
                "vertical": np.sqrt(np.mean(cV ** 2)),
                "diagonal": np.sqrt(np.mean(cD ** 2)),
            }
            features[f"level_{level_idx}"] = level_energy
            energy_per_level.append(sum(level_energy.values()))

        total_energy = sum(energy_per_level)
        energy_distribution = [e / (total_energy + 1e-10) for e in energy_per_level]

        return {
            "wavelet_features": features,
            "energy_per_level": energy_per_level,
            "energy_distribution": energy_distribution,
            "detail_to_approx_ratio": sum(energy_per_level) / (np.sqrt(np.mean(coeffs[0] ** 2)) + 1e-10),
        }

    def compute_power_spectrum_slope(self, image):
        spectrum = self.compute_fft_spectrum(image)
        radial_avg = self.compute_azimuthal_average(spectrum["magnitude_spectrum"])

        freqs = np.arange(1, len(radial_avg) + 1)
        valid = radial_avg > 0
        if valid.sum() < 2:
            return {"slope": 0.0, "intercept": 0.0, "r_squared": 0.0}

        log_freq = np.log(freqs[valid])
        log_power = np.log(radial_avg[valid])

        coefficients = np.polyfit(log_freq, log_power, 1)
        slope, intercept = coefficients

        predicted = slope * log_freq + intercept
        ss_res = np.sum((log_power - predicted) ** 2)
        ss_tot = np.sum((log_power - log_power.mean()) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "radial_profile": radial_avg,
        }

    def analyze(self, image):
        fft_result = self.compute_fft_spectrum(image)
        dct_result = self.compute_dct_statistics(image)
        wavelet_result = self.compute_wavelet_features(image)
        spectral_slope = self.compute_power_spectrum_slope(image)
        radial_avg = self.compute_azimuthal_average(fft_result["magnitude_spectrum"])

        anomaly_indicators = {
            "high_freq_energy": dct_result["high_freq_ratio"],
            "spectral_slope_deviation": abs(spectral_slope["slope"] + 2.0),
            "wavelet_detail_ratio": wavelet_result["detail_to_approx_ratio"],
            "spectral_flatness": radial_avg.std() / (radial_avg.mean() + 1e-10),
        }

        forgery_score = (
            0.3 * min(dct_result["high_freq_ratio"] * 5, 1.0)
            + 0.3 * min(abs(spectral_slope["slope"] + 2.0) / 2.0, 1.0)
            + 0.2 * min(wavelet_result["detail_to_approx_ratio"] / 10.0, 1.0)
            + 0.2 * min(anomaly_indicators["spectral_flatness"] / 2.0, 1.0)
        )

        return {
            "fft": fft_result,
            "dct": dct_result,
            "wavelet": wavelet_result,
            "spectral_slope": spectral_slope,
            "radial_profile": radial_avg,
            "anomaly_indicators": anomaly_indicators,
            "frequency_forgery_score": float(np.clip(forgery_score, 0, 1)),
        }
