import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

from src.analysis.frequency_analysis import FrequencyAnalyzer


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


def test_frequency_analyzer_full(sample_image):
    analyzer = FrequencyAnalyzer()
    result = analyzer.analyze(sample_image)
    assert "fft" in result
    assert "dct" in result
    assert "wavelet" in result
    assert 0 <= result["frequency_forgery_score"] <= 1


def test_fft_spectrum(sample_image):
    analyzer = FrequencyAnalyzer()
    result = analyzer.compute_fft_spectrum(sample_image)
    assert "magnitude_spectrum" in result
    assert "phase_spectrum" in result
    assert result["magnitude_spectrum"].shape == (256, 256)


def test_dct_statistics(sample_image):
    analyzer = FrequencyAnalyzer()
    result = analyzer.compute_dct_statistics(sample_image)
    assert result["dct_energy_map"].shape == (8, 8)
    assert 0 <= result["high_freq_ratio"] <= 1


def test_wavelet_features(sample_image):
    analyzer = FrequencyAnalyzer()
    result = analyzer.compute_wavelet_features(sample_image)
    assert len(result["energy_per_level"]) == 4
    assert len(result["energy_distribution"]) == 4
