import os
from dotenv import load_dotenv

# Load variables from .env if it exists
load_dotenv()

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "face_attendance")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "password")
DB_PORT = os.getenv("DB_PORT", "5432")

# Recognition Configuration
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.5"))
CONFIRMATION_FRAMES = int(os.getenv("CONFIRMATION_FRAMES", "3"))
CHECK_IN_COOLDOWN_MINUTES = int(os.getenv("CHECK_IN_COOLDOWN_MINUTES", "1"))

# File System Configuration
UNKNOWN_FACES_DIR = os.getenv("UNKNOWN_FACES_DIR", "data/unknown/")
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# Camera Configuration
# If CAMERA_SOURCE is numeric, treat it as a camera index; otherwise, it's an RTSP/Video path.
source = os.getenv("CAMERA_SOURCE", "0")
if source.isdigit():
    CAMERA_SOURCE = int(source)
else:
    CAMERA_SOURCE = source

# Web Configuration
flask_port_str = os.getenv("FLASK_PORT", "5000")
FLASK_PORT = int(flask_port_str) if flask_port_str.strip() else 5000

# DeepFace Model Configuration
# Facenet512 matches the 512-dimension vector in our DB schema
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
# Don't raise an error if a face isn't clearly found in every frame
ENFORCE_DETECTION = False
ALIGN = True

# Two-stage detection configuration
def get_env_int(name, default):
    val = os.getenv(name, "")
    return int(val) if val.strip() else default

def get_env_float(name, default):
    val = os.getenv(name, "")
    return float(val) if val.strip() else default

DETECTION_WIDTH = get_env_int("DETECTION_WIDTH", 640)
DETECTION_HEIGHT = get_env_int("DETECTION_HEIGHT", 360)
DISPLAY_WIDTH = get_env_int("DISPLAY_WIDTH", 960)
DISPLAY_HEIGHT = get_env_int("DISPLAY_HEIGHT", 540)
YUNET_SCORE_THRESHOLD = get_env_float("YUNET_SCORE_THRESHOLD", 0.5)
YUNET_NMS_THRESHOLD = get_env_float("YUNET_NMS_THRESHOLD", 0.3)
YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")

# Face crop padding multiplier (1.5 = 50% padding around detected face)
FACE_CROP_PADDING = get_env_float("FACE_CROP_PADDING", 1.5)
