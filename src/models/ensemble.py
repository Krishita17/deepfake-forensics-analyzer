import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficientnet_detector import EfficientNetDetector
from .xception_detector import XceptionDetector
from .capsule_network import CapsuleNetwork
from .frequency_network import FrequencyAwareNetwork
from .attention_module import FeaturePyramidFusion


class UncertaintyEstimator(nn.Module):
    def __init__(self, feature_dim, num_mc_samples=10):
        super().__init__()
        self.num_mc_samples = num_mc_samples
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(feature_dim, 2)

    def forward(self, features):
        if self.training:
            return self.fc(self.dropout(features)), torch.tensor(0.0)

        predictions = []
        for _ in range(self.num_mc_samples):
            pred = F.softmax(self.fc(self.dropout(features)), dim=1)
            predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        epistemic = predictions.var(dim=0).sum(dim=-1)
        aleatoric = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)

        return mean_pred, {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total_uncertainty": epistemic + aleatoric,
        }


class EnsembleDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.models = nn.ModuleDict()
        self.model_weights = {}
        feature_dims = []

        for model_cfg in config["model"]["ensemble"]["models"]:
            name = model_cfg["name"]
            weight = model_cfg["weight"]
            self.model_weights[name] = weight

            if name == "efficientnet_b4":
                self.models[name] = EfficientNetDetector(
                    num_classes=config["model"]["num_classes"],
                    dropout=config["model"]["dropout"],
                )
                feature_dims.append(512)
            elif name == "xception":
                self.models[name] = XceptionDetector(
                    num_classes=config["model"]["num_classes"],
                    dropout=config["model"]["dropout"],
                )
                feature_dims.append(2048)
            elif name == "capsule_net":
                self.models[name] = CapsuleNetwork(
                    num_classes=config["model"]["num_classes"],
                    dropout=config["model"]["dropout"],
                )
                feature_dims.append(32)
            elif name == "frequency_net":
                self.models[name] = FrequencyAwareNetwork(
                    num_classes=config["model"]["num_classes"],
                    dropout=config["model"]["dropout"],
                )
                feature_dims.append(512)

        self.feature_fusion = FeaturePyramidFusion(feature_dims, output_dim=512)
        self.uncertainty_estimator = UncertaintyEstimator(512)

        self.meta_classifier = nn.Sequential(
            nn.Linear(len(self.models) * 2 + 512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        self.confidence_calibrator = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        all_outputs = {}
        all_probs = []
        all_features = []

        for name, model in self.models.items():
            out = model(x)
            all_outputs[name] = out
            all_probs.append(out["probabilities"])
            all_features.append(out["features"])

        fused_features = self.feature_fusion(all_features)

        weighted_probs = sum(
            self.model_weights[name] * out["probabilities"]
            for name, out in all_outputs.items()
        )

        prob_concat = torch.cat(all_probs, dim=1)
        meta_input = torch.cat([prob_concat, fused_features], dim=1)
        meta_logits = self.meta_classifier(meta_input)

        calibrated_logits = self.confidence_calibrator(meta_logits)
        calibrated_probs = F.softmax(calibrated_logits, dim=1)

        mc_pred, uncertainty = self.uncertainty_estimator(fused_features)

        model_agreement = torch.stack(all_probs).std(dim=0).mean(dim=-1)

        return {
            "logits": meta_logits,
            "probabilities": calibrated_probs,
            "weighted_ensemble_probs": weighted_probs,
            "individual_outputs": all_outputs,
            "uncertainty": uncertainty,
            "model_agreement": 1.0 - model_agreement,
            "fused_features": fused_features,
            "confidence": calibrated_probs.max(dim=1).values,
        }
