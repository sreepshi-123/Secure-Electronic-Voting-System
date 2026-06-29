import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
face_detector = MTCNN(
    image_size=160,
    margin=20,
    post_process=True,
    select_largest=True,
    device=device,
)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def normalize_face_image(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def estimate_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray) / 255.0)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness_score = max(0.0, 1.0 - abs(brightness - 0.5) * 2)
    blur_normalized = min(1.0, blur_score / 250.0)
    quality_score = round((brightness_score * 0.4) + (blur_normalized * 0.6), 4)

    return {
        'brightness': round(brightness, 4),
        'blur_score': round(blur_score, 4),
        'quality_score': quality_score,
    }


def get_face_embedding(image):
    normalized = normalize_face_image(image)
    rgb_image = normalized[:, :, ::-1]
    pil_image = Image.fromarray(rgb_image)
    tensor = face_detector(pil_image)

    if tensor is None:
        return None

    with torch.no_grad():
        embedding = model(tensor.unsqueeze(0).to(device)).cpu().numpy()[0]

    return embedding.tolist()


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
