import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 13, 308, 14]

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)


def _landmark_point(landmarks, idx, w, h):
    point = landmarks[idx]
    return np.array([point.x * w, point.y * h], dtype=np.float32)


def _eye_aspect_ratio(points):
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal == 0:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def extract_iris_features(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    h, w, _ = image.shape

    left_points = []
    right_points = []

    for idx in LEFT_IRIS:
        x = landmarks[idx].x
        y = landmarks[idx].y
        left_points.extend([round(float(x), 6), round(float(y), 6)])

    for idx in RIGHT_IRIS:
        x = landmarks[idx].x
        y = landmarks[idx].y
        right_points.extend([round(float(x), 6), round(float(y), 6)])

    left_eye_points = [_landmark_point(landmarks, idx, w, h) for idx in LEFT_EYE]
    right_eye_points = [_landmark_point(landmarks, idx, w, h) for idx in RIGHT_EYE]
    mouth_points = [_landmark_point(landmarks, idx, w, h) for idx in MOUTH]

    left_ear = _eye_aspect_ratio(left_eye_points)
    right_ear = _eye_aspect_ratio(right_eye_points)
    mouth_open_ratio = float(np.linalg.norm(mouth_points[1] - mouth_points[3]) / max(np.linalg.norm(mouth_points[0] - mouth_points[2]), 1e-6))
    iris_quality = round(min(1.0, ((left_ear + right_ear) / 2.0) * 8.0), 4)

    return {
        'left': left_points,
        'right': right_points,
        'quality_score': iris_quality,
        'eye_open_score': round((left_ear + right_ear) / 2.0, 4),
        'mouth_open_score': round(mouth_open_ratio, 4),
    }


def detect_passive_liveness(iris_features):
    if iris_features is None:
        return False, 'Face landmarks not available for liveness checks.'

    eye_open_score = iris_features.get('eye_open_score', 0.0)
    if eye_open_score < 0.18:
        return False, 'Eyes appear closed or not clearly visible. Please keep both eyes open.'

    return True, None


def iris_distance(a, b):
    a_left = np.array(a['left'])
    b_left = np.array(b['left'])
    a_right = np.array(a['right'])
    b_right = np.array(b['right'])

    left_distance = np.linalg.norm(a_left - b_left)
    right_distance = np.linalg.norm(a_right - b_right)
    return float((left_distance + right_distance) / 2.0)
