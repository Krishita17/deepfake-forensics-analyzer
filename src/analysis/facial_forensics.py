import numpy as np
import cv2
from scipy.stats import entropy


class FacialForensicsAnalyzer:
    def __init__(self, landmark_detector="opencv"):
        self.detector_type = landmark_detector
        self._face_cascade = None
        self._landmark_model = None

    def _init_detector(self):
        if self._face_cascade is None:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        if self._landmark_model is None:
            self._landmark_model = cv2.face.createFacemarkLBF() if hasattr(cv2, "face") else None

    def detect_face_region(self, image):
        self._init_detector()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return {"box": (x, y, w, h), "center": (x + w // 2, y + h // 2)}

    def _generate_pseudo_landmarks(self, image, face_box):
        x, y, w, h = face_box
        landmarks = np.zeros((68, 3), dtype=np.float64)

        landmarks[0] = [x, y + h * 0.45, 0]
        landmarks[16] = [x + w, y + h * 0.45, 0]
        for i in range(1, 16):
            t = i / 16.0
            lx = x + t * w
            ly = y + h * (0.45 + 0.55 * np.sin(t * np.pi) * 0.3)
            landmarks[i] = [lx, ly, 0]

        landmarks[36] = [x + w * 0.25, y + h * 0.35, 0]
        landmarks[39] = [x + w * 0.40, y + h * 0.35, 0]
        landmarks[42] = [x + w * 0.60, y + h * 0.35, 0]
        landmarks[45] = [x + w * 0.75, y + h * 0.35, 0]

        landmarks[30] = [x + w * 0.50, y + h * 0.55, 0]
        landmarks[33] = [x + w * 0.50, y + h * 0.60, 0]

        landmarks[48] = [x + w * 0.32, y + h * 0.72, 0]
        landmarks[54] = [x + w * 0.68, y + h * 0.72, 0]
        landmarks[51] = [x + w * 0.50, y + h * 0.70, 0]
        landmarks[57] = [x + w * 0.50, y + h * 0.78, 0]

        landmarks[27] = [x + w * 0.50, y + h * 0.20, 0]

        return landmarks

    def compute_landmark_consistency(self, landmarks):
        if landmarks is None:
            return {"score": 0.0, "details": "no_face_detected",
                    "symmetry_score": 0.0, "proportion_deviation": 0.0,
                    "jaw_smoothness": 0.0, "eye_distance": 0.0, "consistency_score": 0.0}

        left_eye = landmarks[36]
        right_eye = landmarks[45]
        nose_tip = landmarks[30]
        left_mouth = landmarks[48]
        right_mouth = landmarks[54]

        eye_dist = np.linalg.norm(left_eye[:2] - right_eye[:2])
        left_nose = np.linalg.norm(left_eye[:2] - nose_tip[:2])
        right_nose = np.linalg.norm(right_eye[:2] - nose_tip[:2])
        mouth_width = np.linalg.norm(left_mouth[:2] - right_mouth[:2])

        symmetry_score = 1.0 - abs(left_nose - right_nose) / (eye_dist + 1e-10)
        proportion_score = mouth_width / (eye_dist + 1e-10)
        expected_proportion = 0.65
        proportion_deviation = abs(proportion_score - expected_proportion)

        jaw_points = landmarks[0:17]
        if len(jaw_points) >= 3:
            diffs = np.diff(jaw_points[:, :2], axis=0)
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            angle_smoothness = np.std(np.diff(angles))
        else:
            angle_smoothness = 0.0

        return {
            "symmetry_score": float(np.clip(symmetry_score, 0, 1)),
            "proportion_deviation": float(proportion_deviation),
            "jaw_smoothness": float(angle_smoothness),
            "eye_distance": float(eye_dist),
            "consistency_score": float(np.clip(
                0.4 * symmetry_score
                + 0.3 * (1.0 - min(proportion_deviation * 5, 1.0))
                + 0.3 * (1.0 - min(angle_smoothness * 10, 1.0)),
                0, 1,
            )),
        }

    def analyze_skin_texture(self, image, face_box):
        if face_box is None:
            return {"score": 0.0, "texture_variance": 0.0,
                    "gabor_uniformity": 0.0, "color_consistency": 0.0, "texture_score": 0.0}

        h, w = image.shape[:2]
        x, y, bw, bh = face_box
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = x + bw // 2, y + bh // 2
        cv2.ellipse(mask, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 255, -1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_region = cv2.bitwise_and(gray, gray, mask=mask)

        laplacian = cv2.Laplacian(face_region, cv2.CV_64F)
        texture_variance = laplacian[mask > 0].var() if (mask > 0).any() else 0

        gabor_responses = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            gabor_responses.append(filtered[mask > 0].mean() if (mask > 0).any() else 0)

        gabor_uniformity = 1.0 - (np.std(gabor_responses) / (np.mean(gabor_responses) + 1e-10))

        left_cheek = (x + bw // 4, y + bh // 2)
        right_cheek = (x + 3 * bw // 4, y + bh // 2)
        forehead = (x + bw // 2, y + bh // 4)

        def get_patch_stats(center, size=30):
            px, py = int(center[0]), int(center[1])
            y1, y2 = max(0, py - size), min(h, py + size)
            x1, x2 = max(0, px - size), min(w, px + size)
            patch = gray[y1:y2, x1:x2]
            if patch.size == 0:
                return 0, 0
            return float(patch.mean()), float(patch.std())

        patches = [get_patch_stats(left_cheek), get_patch_stats(right_cheek), get_patch_stats(forehead)]
        mean_diffs = [abs(patches[i][0] - patches[j][0]) for i in range(len(patches)) for j in range(i+1, len(patches))]
        color_consistency = 1.0 - min(np.mean(mean_diffs) / 50.0, 1.0)

        return {
            "texture_variance": float(texture_variance),
            "gabor_uniformity": float(np.clip(gabor_uniformity, 0, 1)),
            "color_consistency": float(np.clip(color_consistency, 0, 1)),
            "texture_score": float(np.clip(
                0.4 * min(texture_variance / 500.0, 1.0)
                + 0.3 * gabor_uniformity
                + 0.3 * color_consistency,
                0, 1,
            )),
        }

    def detect_blending_artifacts(self, image, face_box):
        if face_box is None:
            return {"score": 0.0, "edge_density_at_boundary": 0.0,
                    "color_discontinuity": 0.0, "blending_score": 0.0}

        h, w = image.shape[:2]
        x, y, bw, bh = face_box
        cx, cy = x + bw // 2, y + bh // 2

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
        boundary = cv2.dilate(mask, np.ones((5, 5)), iterations=3) - cv2.erode(mask, np.ones((5, 5)), iterations=3)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        boundary_edges = cv2.bitwise_and(edges, edges, mask=boundary)

        edge_density = boundary_edges.sum() / (boundary.sum() + 1e-10)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        inner_mask = cv2.erode(mask, np.ones((10, 10)), iterations=2)
        outer_mask = cv2.dilate(mask, np.ones((10, 10)), iterations=2) - mask

        inner_vals = lab[inner_mask > 0].mean(axis=0) if (inner_mask > 0).any() else np.zeros(3)
        outer_vals = lab[outer_mask > 0].mean(axis=0) if (outer_mask > 0).any() else np.zeros(3)
        color_discontinuity = np.linalg.norm(inner_vals - outer_vals) / 100.0

        return {
            "edge_density_at_boundary": float(edge_density),
            "color_discontinuity": float(color_discontinuity),
            "blending_score": float(np.clip(
                0.5 * min(edge_density * 20, 1.0)
                + 0.5 * min(color_discontinuity * 3, 1.0),
                0, 1,
            )),
        }

    def analyze(self, image):
        face_info = self.detect_face_region(image)
        face_box = face_info["box"] if face_info else None

        if face_box is not None:
            landmarks = self._generate_pseudo_landmarks(image, face_box)
        else:
            landmarks = None

        consistency = self.compute_landmark_consistency(landmarks)
        texture = self.analyze_skin_texture(image, face_box)
        blending = self.detect_blending_artifacts(image, face_box)

        if landmarks is None:
            return {
                "landmarks": None,
                "consistency": consistency,
                "texture": texture,
                "blending": blending,
                "facial_forgery_score": 0.5,
            }

        facial_score = (
            0.35 * (1.0 - consistency["consistency_score"])
            + 0.30 * (1.0 - texture["texture_score"])
            + 0.35 * blending["blending_score"]
        )

        return {
            "landmarks": landmarks,
            "consistency": consistency,
            "texture": texture,
            "blending": blending,
            "facial_forgery_score": float(np.clip(facial_score, 0, 1)),
        }
