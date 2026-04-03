import os
from dotenv import load_dotenv

load_dotenv()


def _env(name, default, cast=str):
    """Read an env var, returning *default* when the var is missing or blank."""
    val = os.getenv(name, "")
    return cast(val) if val.strip() else default


# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "face_attendance")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "password")
DB_PORT = os.getenv("DB_PORT", "5432")

# Recognition
RECOGNITION_THRESHOLD = _env("RECOGNITION_THRESHOLD", 0.5, float)
CONFIRMATION_FRAMES = _env("CONFIRMATION_FRAMES", 3, int)
CHECK_IN_COOLDOWN_MINUTES = _env("CHECK_IN_COOLDOWN_MINUTES", 1, int)

# File system
UNKNOWN_FACES_DIR = os.getenv("UNKNOWN_FACES_DIR", "data/unknown/")
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
MEMBER_FACES_DIR = os.getenv("MEMBER_FACES_DIR", "data/members/")
os.makedirs(MEMBER_FACES_DIR, exist_ok=True)

# Camera — numeric values are webcam indices, anything else is an RTSP/video path
_camera_source = os.getenv("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(_camera_source) if _camera_source.isdigit() else _camera_source

# Web
FLASK_PORT = _env("FLASK_PORT", 5000, int)

# DeepFace — Facenet512 matches the 512-dimension vector in our DB schema
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = False
ALIGN = True

# Two-stage detection
DETECTION_WIDTH = _env("DETECTION_WIDTH", 640, int)
DETECTION_HEIGHT = _env("DETECTION_HEIGHT", 360, int)
DISPLAY_WIDTH = _env("DISPLAY_WIDTH", 960, int)
DISPLAY_HEIGHT = _env("DISPLAY_HEIGHT", 540, int)
YUNET_SCORE_THRESHOLD = _env("YUNET_SCORE_THRESHOLD", 0.5, float)
YUNET_NMS_THRESHOLD = _env("YUNET_NMS_THRESHOLD", 0.3, float)
YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")

# Face crop padding multiplier (1.5 = 50% padding around detected face)
FACE_CROP_PADDING = _env("FACE_CROP_PADDING", 1.5, float)
