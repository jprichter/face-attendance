import os
import signal
from datetime import datetime

import cv2
import numpy as np
import psycopg2
from deepface import DeepFace

import config
from logger import log_check_in, log_unknown_detection

FACE_CONFIDENCE_THRESHOLD = 0.6

running = True


def signal_handler(sig, frame):
    global running
    print("\nStopping monitoring...")
    running = False


def get_connection():
    try:
        return psycopg2.connect(
            host=config.DB_HOST,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASS,
            port=config.DB_PORT,
            connect_timeout=5,
        )
    except Exception:
        return None


def normalize_frame(frame):
    """Apply histogram equalization to the luminance channel."""
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv_frame[:, :, 0] = cv2.equalizeHist(yuv_frame[:, :, 0])
    return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)


def query_database(embedding):
    """Find the closest member by cosine distance. Returns (member_id, name, distance)."""
    conn = get_connection()
    if not conn:
        return None, "DB Error", None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, face_embedding <=> %s::vector AS distance "
            "FROM members ORDER BY distance ASC LIMIT 1",
            (list(embedding),),
        )
        result = cur.fetchone()
        if result and result[2] < config.RECOGNITION_THRESHOLD:
            return result[0], result[1], result[2]
        return None, "Unknown", result[2] if result else None
    except Exception as e:
        print(f"Database query error: {e}")
        return None, "Error", None
    finally:
        conn.close()


def extract_embedding(image):
    """Run DeepFace extract + represent on an image. Returns (embedding, face_data) or None."""
    faces = DeepFace.extract_faces(
        img_path=image,
        detector_backend=config.DETECTOR_BACKEND,
        align=config.ALIGN,
        enforce_detection=False,
    )

    for face_data in faces:
        if face_data["confidence"] < FACE_CONFIDENCE_THRESHOLD:
            continue

        results = DeepFace.represent(
            img_path=face_data["face"],
            model_name=config.MODEL_NAME,
            detector_backend="skip",
            align=False,
            enforce_detection=False,
        )

        if results:
            return results[0]["embedding"], face_data

    return None


def handle_recognition(embedding, confirmation_buffer, image_to_save):
    """Query database and update confirmation buffer. Returns the matched name."""
    member_id, name, distance = query_database(embedding)

    if member_id:
        if confirmation_buffer["id"] == member_id:
            confirmation_buffer["count"] += 1
        else:
            confirmation_buffer["id"] = member_id
            confirmation_buffer["count"] = 1

        if confirmation_buffer["count"] >= config.CONFIRMATION_FRAMES:
            print(f"Confirmed match: {name} (Dist: {distance:.4f})")
            log_check_in(member_id)
            confirmation_buffer["count"] = 0

    elif distance is not None and distance >= config.RECOGNITION_THRESHOLD:
        print(f"Unknown face detected (Distance: {distance:.4f})")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(config.UNKNOWN_FACES_DIR, f"unknown_{timestamp}.jpg")
        cv2.imwrite(image_path, image_to_save)
        log_unknown_detection(embedding, image_path)
        confirmation_buffer["id"] = None
        confirmation_buffer["count"] = 0

    return name


def cleanup_unknown():
    """Prompt user to empty the unknown faces directory."""
    count = len([f for f in os.listdir(config.UNKNOWN_FACES_DIR) if f.endswith(".jpg")])
    if count == 0:
        return

    print(f"\nThere are {count} unknown face captures in {config.UNKNOWN_FACES_DIR}.")
    choice = input("Would you like to clear these images? (y/n): ").strip().lower()
    if choice == "y":
        for filename in os.listdir(config.UNKNOWN_FACES_DIR):
            file_path = os.path.join(config.UNKNOWN_FACES_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print("Unknown faces directory cleared.")


def ensure_yunet_model():
    """Download YuNet model if not present."""
    os.makedirs(os.path.dirname(config.YUNET_MODEL_PATH), exist_ok=True)
    if not os.path.exists(config.YUNET_MODEL_PATH):
        import urllib.request

        url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        print("Downloading YuNet model...")
        urllib.request.urlretrieve(url, config.YUNET_MODEL_PATH)
        print(f"Downloaded to {config.YUNET_MODEL_PATH}")


def create_yunet_detector():
    """Create and configure YuNet face detector."""
    ensure_yunet_model()
    return cv2.FaceDetectorYN.create(
        config.YUNET_MODEL_PATH,
        "",
        (config.DETECTION_WIDTH, config.DETECTION_HEIGHT),
        score_threshold=config.YUNET_SCORE_THRESHOLD,
        nms_threshold=config.YUNET_NMS_THRESHOLD,
        top_k=5000,
    )


def map_face_to_original(face, det_w, det_h, orig_w, orig_h):
    """Map YuNet detection from small frame to original coordinates with padding."""
    x, y, w, h = face[0], face[1], face[2], face[3]
    scale_x = orig_w / det_w
    scale_y = orig_h / det_h

    pad = config.FACE_CROP_PADDING
    cx, cy = x + w / 2, y + h / 2
    crop_w, crop_h = w * pad, h * pad

    x1 = int(max(0, (cx - crop_w / 2) * scale_x))
    y1 = int(max(0, (cy - crop_h / 2) * scale_y))
    x2 = int(min(orig_w, (cx + crop_w / 2) * scale_x))
    y2 = int(min(orig_h, (cy + crop_h / 2) * scale_y))

    return x1, y1, x2, y2


def process_frame_single_stage(frame, confirmation_buffer):
    """Single-stage pipeline for low-res sources (webcam). DeepFace handles detection + embedding."""
    processed_frame = normalize_frame(frame)

    try:
        result = extract_embedding(processed_frame)
        if result is None:
            return frame

        embedding, face_data = result
        name = handle_recognition(embedding, confirmation_buffer, frame)

        region = face_data["facial_area"]
        cv2.rectangle(
            frame,
            (region["x"], region["y"]),
            (region["x"] + region["w"], region["y"] + region["h"]),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame, name, (region["x"], region["y"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )

    except Exception as e:
        print(f"Face processing error: {e}")

    return frame


def process_frame_two_stage(frame, yunet, confirmation_buffer):
    """Two-stage pipeline for high-res sources (4K RTSP). YuNet detects, DeepFace embeds."""
    orig_h, orig_w = frame.shape[:2]

    detection_frame = cv2.resize(frame, (config.DETECTION_WIDTH, config.DETECTION_HEIGHT))
    detection_frame = normalize_frame(detection_frame)

    display_frame = cv2.resize(frame, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
    display_scale_x = config.DISPLAY_WIDTH / orig_w
    display_scale_y = config.DISPLAY_HEIGHT / orig_h

    _, faces_detected = yunet.detect(detection_frame)

    if faces_detected is None:
        return display_frame

    for face in faces_detected:
        try:
            x1, y1, x2, y2 = map_face_to_original(
                face, config.DETECTION_WIDTH, config.DETECTION_HEIGHT, orig_w, orig_h
            )
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            result = extract_embedding(face_crop)
            if result is None:
                continue

            embedding, _ = result
            name = handle_recognition(embedding, confirmation_buffer, face_crop)

            dx1 = int(x1 * display_scale_x)
            dy1 = int(y1 * display_scale_y)
            dx2 = int(x2 * display_scale_x)
            dy2 = int(y2 * display_scale_y)
            cv2.rectangle(display_frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
            cv2.putText(
                display_frame, name, (dx1, dy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        except Exception as e:
            print(f"Face processing error: {e}")

    return display_frame


def main():
    global running
    running = True
    signal.signal(signal.SIGINT, signal_handler)

    print(f"\n--- Face Attendance Monitoring ---")
    print(f"Log Viewer available at: http://localhost:{config.FLASK_PORT}")
    print(f"Press 'q' in the camera window or Ctrl+C to exit.\n")

    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open camera {config.CAMERA_SOURCE}.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return

    orig_h, orig_w = frame.shape[:2]
    use_two_stage = orig_w > config.DETECTION_WIDTH

    yunet = None
    if use_two_stage:
        yunet = create_yunet_detector()
        print(f"High-res source ({orig_w}x{orig_h}) — using two-stage YuNet pipeline")
    else:
        print(f"Low-res source ({orig_w}x{orig_h}) — using single-stage pipeline")

    confirmation_buffer = {"id": None, "count": 0}

    while running:
        if use_two_stage:
            display = process_frame_two_stage(frame, yunet, confirmation_buffer)
        else:
            display = process_frame_single_stage(frame, confirmation_buffer)

        cv2.imshow("Face Attendance Monitoring", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

    cap.release()
    cv2.destroyAllWindows()
    cleanup_unknown()


if __name__ == "__main__":
    main()
