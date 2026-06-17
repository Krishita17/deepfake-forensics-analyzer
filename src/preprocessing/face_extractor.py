import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch


class FaceExtractor:
    def __init__(self, margin=0.3, min_face_size=60, device=None):
        self.margin = margin
        self.min_face_size = min_face_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=380,
            margin=int(margin * 100),
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=self.device,
            keep_all=True,
        )

    def extract_faces(self, image, return_landmarks=False):
        if isinstance(image, np.ndarray):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.array(image)

        boxes, probs, landmarks = self.mtcnn.detect(rgb, landmarks=True)

        if boxes is None:
            return []

        faces = []
        h, w = rgb.shape[:2]

        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < 0.9:
                continue

            x1, y1, x2, y2 = box
            face_w = x2 - x1
            face_h = y2 - y1
            margin_x = face_w * self.margin
            margin_y = face_h * self.margin

            x1 = max(0, int(x1 - margin_x))
            y1 = max(0, int(y1 - margin_y))
            x2 = min(w, int(x2 + margin_x))
            y2 = min(h, int(y2 + margin_y))

            face_crop = rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            result = {
                "face": face_crop,
                "box": [x1, y1, x2, y2],
                "confidence": float(prob),
            }
            if return_landmarks and landmarks is not None:
                result["landmarks"] = landmarks[i]

            faces.append(result)

        faces.sort(key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]), reverse=True)
        return faces

    def extract_from_video(self, video_path, sample_rate=10, max_faces_per_frame=1):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_faces = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                faces = self.extract_faces(frame)
                for face_data in faces[:max_faces_per_frame]:
                    face_data["frame_idx"] = frame_idx
                    face_data["timestamp"] = frame_idx / fps
                    all_faces.append(face_data)

            frame_idx += 1

        cap.release()
        return all_faces, {"fps": fps, "total_frames": total_frames}
