import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class JPEGCompression(A.ImageOnlyTransform):
    def __init__(self, quality_range=(30, 95), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.quality_range = quality_range

    def apply(self, img, quality=70, **params):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", img, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def get_params(self):
        return {"quality": np.random.randint(*self.quality_range)}


class DownscaleUpscale(A.ImageOnlyTransform):
    def __init__(self, scale_range=(0.25, 0.75), always_apply=False, p=0.3):
        super().__init__(always_apply, p)
        self.scale_range = scale_range

    def apply(self, img, scale=0.5, **params):
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def get_params(self):
        return {"scale": np.random.uniform(*self.scale_range)}


def get_training_transforms(image_size=380):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=(3, 7), p=1),
            A.MedianBlur(blur_limit=5, p=1),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.CLAHE(clip_limit=4.0, p=1),
        ], p=0.3),
        JPEGCompression(quality_range=(30, 95), p=0.5),
        DownscaleUpscale(scale_range=(0.25, 0.75), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(10, 30), hole_width_range=(10, 30), p=0.2),
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_transforms(image_size=380):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
