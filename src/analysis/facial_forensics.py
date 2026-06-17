import numpy as np
import cv2
from scipy.spatial import ConvexHull
from scipy.stats import entropy


class FacialForensicsAnalyzer:
    def __init__(self, landmark_detector="mediapipe"):
        self.detector_type = landmark_detector
        self._detector = None
        self._face_mesh = None

    def _init_mediapipe(self):
        if self._face_mesh is None:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

    def detect_landmarks(self, image):
        self._init_mediapipe()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        points = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in landmarks.landmark
        ])
        return points

    def compute_landmark_consistency(self, landmarks):
        if landmarks is None:
            return {"score": 0.0, "details": "no_face_detected"}

        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        eye_dist = np.linalg.norm(left_eye[:2] - right_eye[:2])
        left_nose = np.linalg.norm(left_eye[:2] - nose_tip[:2])
        right_nose = np.linalg.norm(right_eye[:2] - nose_tip[:2])
        mouth_width = np.linalg.norm(left_mouth[:2] - right_mouth[:2])

        symmetry_score = 1.0 - abs(left_nose - right_nose) / (eye_dist + 1e-10)
        proportion_score = mouth_width / (eye_dist + 1e-10)
        expected_proportion = 0.65
        proportion_deviation = abs(proportion_score - expected_proportion)

        jaw_points = landmarks[152:172] if len(landmarks) > 172 else landmarks[-20:]
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

    def analyze_skin_texture(self, image, landmarks):
        if landmarks is None:
            return {"score": 0.0}

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                             323, 361, 288, 397, 365, 379, 378, 400, 377,
                             152, 148, 176, 149, 150, 136, 172, 58, 132,
                             93, 234, 127, 162, 21, 54, 103, 67, 109]
        pts = landmarks[face_oval_indices, :2].astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255)

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

        left_cheek = landmarks[234, :2].astype(int)
        right_cheek = landmarks[454, :2].astype(int)
        forehead = landmarks[10, :2].astype(int)

        def get_patch_stats(center, size=30):
            y, x = int(center[1]), int(center[0])
            y1, y2 = max(0, y - size), min(h, y + size)
            x1, x2 = max(0, x - size), min(w, x + size)
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

    def detect_blending_artifacts(self, image, landmarks):
        if landmarks is None:
            return {"score": 0.0}

        h, w = image.shape[:2]
        face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                             323, 361, 288, 397, 365, 379, 378, 400, 377,
                             152, 148, 176, 149, 150, 136, 172, 58, 132,
                             93, 234, 127, 162, 21, 54, 103, 67, 109]
        pts = landmarks[face_oval_indices, :2].astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 255)
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
        landmarks = self.detect_landmarks(image)
        consistency = self.compute_landmark_consistency(landmarks)
        texture = self.analyze_skin_texture(image, landmarks)
        blending = self.detect_blending_artifacts(image, landmarks)

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
