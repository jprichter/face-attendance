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
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

# DeepFace Model Configuration
MODEL_NAME = "Facenet512" # Matches the 512-dimension vector in our DB schema
DETECTOR_BACKEND = "opencv"
ENFORCE_DETECTION = False # Don't raise an error if a face isn't clearly found in every frame
ALIGN = True
