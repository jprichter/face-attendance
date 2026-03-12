import psycopg2
from datetime import datetime, timedelta
import config

def get_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        database=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASS,
        port=config.DB_PORT
    )

def log_check_in(member_id):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Check for cool-down
        cooldown_time = datetime.now() - timedelta(minutes=config.CHECK_IN_COOLDOWN_MINUTES)
        cur.execute(
            "SELECT id FROM attendance_log WHERE member_id = %s AND check_in_time > %s LIMIT 1",
            (member_id, cooldown_time)
        )
        already_logged = cur.fetchone()

        if already_logged:
            print(f"Skipping check-in for member {member_id} due to cool-down.")
            return False

        # Perform check-in
        cur.execute(
            "INSERT INTO attendance_log (member_id) VALUES (%s)",
            (member_id,)
        )
        conn.commit()
        print(f"Successfully logged check-in for member {member_id}.")
        return True

    except Exception as e:
        print(f"Database error during check-in: {e}")
        return False
    finally:
        if conn:
            conn.close()

def log_unknown_detection(embedding, image_path):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Convert list/numpy array embedding to list for pgvector (it accepts list format as [1,2,...])
        embedding_list = list(embedding)
        
        cur.execute(
            "INSERT INTO unknown_detections (face_embedding, image_path) VALUES (%s, %s)",
            (embedding_list, image_path)
        )
        conn.commit()
        print(f"Logged unknown detection at {image_path}.")
        return True
    except Exception as e:
        print(f"Error logging unknown detection: {e}")
        return False
    finally:
        if conn:
            conn.close()
