import cv2
import psycopg2
import signal
import sys
import os
import shutil
from deepface import DeepFace
import config
from logger import log_check_in, log_unknown_detection
from datetime import datetime
import numpy as np

# Global flag for exit
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
            connect_timeout=5
        )
    except:
        return None

def normalize_frame(frame):
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv_frame[:, :, 0] = cv2.equalizeHist(yuv_frame[:, :, 0])
    return cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)

def query_database(embedding):
    conn = get_connection()
    if not conn:
        return None, "DB Error", None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, name, face_embedding <=> %s::vector AS distance "
            "FROM members ORDER BY distance ASC LIMIT 1",
            (list(embedding),)
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

def cleanup_unknown():
    """Prompt user to empty the unknown faces directory."""
    count = len([f for f in os.listdir(config.UNKNOWN_FACES_DIR) if f.endswith('.jpg')])
    if count == 0:
        return
    
    print(f"\nThere are {count} unknown face captures in {config.UNKNOWN_FACES_DIR}.")
    choice = input("Would you like to clear these images? (y/n): ").strip().lower()
    if choice == 'y':
        for filename in os.listdir(config.UNKNOWN_FACES_DIR):
            file_path = os.path.join(config.UNKNOWN_FACES_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        print("Unknown faces directory cleared.")

def main():
    global running
    # Enhancement: Graceful exit via Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    print(f"\n--- Face Attendance Monitoring ---")
    print(f"Log Viewer available at: http://localhost:{config.FLASK_PORT}")
    print(f"Press 'q' in the camera window or Ctrl+C to exit.\n")

    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open camera {config.CAMERA_SOURCE}.")
        return

    confirmation_buffer = {"id": None, "count": 0}

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        processed_frame = normalize_frame(frame)

        try:
            faces = DeepFace.extract_faces(
                img_path=processed_frame,
                detector_backend=config.DETECTOR_BACKEND,
                align=config.ALIGN,
                enforce_detection=False
            )

            for face_data in faces:
                if face_data['confidence'] < 0.6:
                    continue

                face_img = face_data['face']
                results = DeepFace.represent(
                    img_path=face_img,
                    model_name=config.MODEL_NAME,
                    detector_backend="skip",
                    align=False,
                    enforce_detection=False
                )
                
                if not results:
                    continue
                
                embedding = results[0]["embedding"]
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
                    image_filename = f"unknown_{timestamp}.jpg"
                    image_path = os.path.join(config.UNKNOWN_FACES_DIR, image_filename)
                    cv2.imwrite(image_path, frame)
                    log_unknown_detection(embedding, image_path)
                    confirmation_buffer["id"] = None
                    confirmation_buffer["count"] = 0

                # Feedback overlay
                region = face_data['facial_area']
                cv2.rectangle(frame, (region['x'], region['y']), (region['x']+region['w'], region['y']+region['h']), (0, 255, 0), 2)
                cv2.putText(frame, f"{name}", (region['x'], region['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        except:
            pass

        cv2.imshow("Face Attendance Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    cap.release()
    cv2.destroyAllWindows()
    
    # Enhancement: Cleanup prompt
    cleanup_unknown()

if __name__ == "__main__":
    main()
