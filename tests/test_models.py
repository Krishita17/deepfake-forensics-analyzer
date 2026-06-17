import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.models.efficientnet_detector import EfficientNetDetector
from src.models.xception_detector import XceptionDetector
from src.models.capsule_network import CapsuleNetwork, MarginLoss
from src.models.frequency_network import FrequencyAwareNetwork


@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 380, 380)


def test_efficientnet_forward(dummy_input):
    model = EfficientNetDetector(num_classes=2, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output["logits"].shape == (2, 2)
    assert output["probabilities"].shape == (2, 2)
    assert output["anomaly_score"].shape == (2,)


def test_xception_forward(dummy_input):
    model = XceptionDetector(num_classes=2)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output["logits"].shape == (2, 2)
    assert output["manipulation_type"].shape == (2, 5)


def test_capsule_network_forward():
    x = torch.randn(2, 3, 128, 128)
    model = CapsuleNetwork(num_classes=2)
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert output["logits"].shape == (2, 2)
    assert output["reconstruction"].shape == (2, 3, 64, 64)


def test_frequency_network_forward(dummy_input):
    model = FrequencyAwareNetwork(num_classes=2)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output["logits"].shape == (2, 2)
    assert output["frequency_anomaly_score"].shape == (2,)


def test_margin_loss():
    loss_fn = MarginLoss()
    lengths = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    targets = torch.tensor([0, 1])
    loss = loss_fn(lengths, targets)
    assert loss.item() >= 0
