import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAMExplainer:
    def __init__(self, model, target_layer_name="features.7"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self._hooks.append(
                    module.register_forward_hook(self._save_activation)
                )
                self._hooks.append(
                    module.register_full_backward_hook(self._save_gradient)
                )
                break

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        logits = output["logits"] if isinstance(output, dict) else output

        if target_class is None:
            target_class = logits.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        logits.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam, size=input_tensor.shape[2:],
            mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def generate_guided_gradcam(self, input_tensor, target_class=None):
        cam = self.generate_heatmap(input_tensor, target_class)

        guided_grads = input_tensor.grad
        if guided_grads is not None:
            guided_grads = guided_grads.squeeze().cpu().numpy()
            guided_grads = np.maximum(guided_grads, 0)
            guided_grads = guided_grads.transpose(1, 2, 0)
            guided_cam = guided_grads * cam[..., np.newaxis]
        else:
            guided_cam = cam

        return cam, guided_cam

    def overlay_heatmap(self, image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

        if image.shape[:2] != heatmap.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))

        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
        return overlay

    def generate_multi_scale_cam(self, input_tensor, scales=[1.0, 0.75, 0.5]):
        cams = []
        original_size = input_tensor.shape[2:]

        for scale in scales:
            if scale != 1.0:
                size = (int(original_size[0] * scale), int(original_size[1] * scale))
                scaled_input = F.interpolate(input_tensor, size=size, mode="bilinear", align_corners=False)
            else:
                scaled_input = input_tensor

            cam = self.generate_heatmap(scaled_input)

            if cam.shape != original_size:
                cam = cv2.resize(cam, (original_size[1], original_size[0]))

            cams.append(cam)

        fused_cam = np.mean(cams, axis=0)
        fused_cam = (fused_cam - fused_cam.min()) / (fused_cam.max() - fused_cam.min() + 1e-8)
        return fused_cam

    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
