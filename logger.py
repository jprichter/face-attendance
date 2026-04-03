import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
import psycopg2

import config

_SCHEMA_READY = False


def get_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        database=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASS,
        port=config.DB_PORT,
    )


def ensure_schema(conn):
    """Apply non-destructive schema upgrades required by newer app features."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return

    cur = conn.cursor()
    try:
        cur.execute(
            "ALTER TABLE members "
            "ADD COLUMN IF NOT EXISTS image_path TEXT"
        )
        cur.execute(
            "ALTER TABLE unknown_detections "
            "ADD COLUMN IF NOT EXISTS image_path TEXT"
        )
        cur.execute(
            "ALTER TABLE unknown_detections "
            "ADD COLUMN IF NOT EXISTS group_id UUID"
        )
        conn.commit()
        _SCHEMA_READY = True
    finally:
        cur.close()


@contextmanager
def _db_cursor(*, commit=False):
    """Yield a DB cursor, optionally committing on success.

    Rolls back on exception, always closes the connection.
    """
    conn = get_connection()
    ensure_schema(conn)
    try:
        cur = conn.cursor()
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_archive_table():
    """Create attendance_archive table if it doesn't exist."""
    try:
        with _db_cursor(commit=True) as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS attendance_archive ("
                "id INTEGER PRIMARY KEY, "
                "member_id INTEGER REFERENCES members(id) ON DELETE CASCADE, "
                "check_in_time TIMESTAMP, "
                "archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
    except Exception as exc:
        print(f"Error creating archive table: {exc}")


def log_check_in(member_id):
    try:
        with _db_cursor(commit=True) as cur:
            cooldown_time = datetime.now() - timedelta(minutes=config.CHECK_IN_COOLDOWN_MINUTES)
            cur.execute(
                "SELECT id FROM attendance_log WHERE member_id = %s AND check_in_time > %s LIMIT 1",
                (member_id, cooldown_time),
            )

            if cur.fetchone():
                print(f"Skipping check-in for member {member_id} due to cool-down.")
                return False

            cur.execute(
                "INSERT INTO attendance_log (member_id) VALUES (%s)",
                (member_id,),
            )
            print(f"Successfully logged check-in for member {member_id}.")
            return True
    except Exception as e:
        print(f"Database error during check-in: {e}")
        return False


def save_member_image(member_id, image_path):
    """Save member photo path, only if they don't already have one."""
    try:
        with _db_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE members SET image_path = %s WHERE id = %s AND image_path IS NULL",
                (image_path, member_id),
            )
            return cur.rowcount > 0
    except Exception as e:
        print(f"Error saving member image: {e}")
        return False


def log_unknown_detection(embedding, image_path, group_id=None):
    try:
        with _db_cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO unknown_detections (face_embedding, image_path, group_id) VALUES (%s, %s, %s)",
                (list(embedding), image_path, group_id),
            )
            print(f"Logged unknown detection at {image_path}.")
            return True
    except Exception as e:
        print(f"Error logging unknown detection: {e}")
        return False


def find_matching_unknown_group(embedding):
    """Find an existing unknown group that matches this embedding."""
    try:
        with _db_cursor() as cur:
            cur.execute(
                "SELECT group_id, face_embedding <=> %s::vector AS distance "
                "FROM unknown_detections "
                "WHERE group_id IS NOT NULL "
                "ORDER BY distance ASC LIMIT 1",
                (list(embedding),),
            )
            result = cur.fetchone()
            if result and result[1] < config.RECOGNITION_THRESHOLD:
                return result[0]
            return None
    except Exception as e:
        print(f"Error finding unknown group: {e}")
        return None


def get_unknown_groups():
    """Get summary of unknown face groups for the dashboard."""
    try:
        with _db_cursor() as cur:
            cur.execute(
                "SELECT ud.group_id, MIN(ud.image_path) AS image_path, "
                "COUNT(*) AS seen_count, "
                "MIN(ud.detected_at) AS first_seen, "
                "MAX(ud.detected_at) AS last_seen "
                "FROM unknown_detections ud "
                "WHERE ud.group_id IS NOT NULL "
                "GROUP BY ud.group_id "
                "ORDER BY MAX(ud.detected_at) DESC"
            )
            return cur.fetchall()
    except Exception as e:
        print(f"Error fetching unknown groups: {e}")
        return []


def get_attendance_logs(limit=100):
    """Fetch current attendance — members checked in within the session window."""
    try:
        with _db_cursor() as cur:
            cutoff = datetime.now() - timedelta(minutes=config.SESSION_DURATION_MINUTES)
            cur.execute(
                "SELECT m.image_path, m.name, a.check_in_time "
                "FROM attendance_log a "
                "JOIN members m ON a.member_id = m.id "
                "WHERE a.check_in_time > %s "
                "ORDER BY a.check_in_time DESC LIMIT %s",
                (cutoff, limit),
            )
            return cur.fetchall()
    except Exception as exc:
        print(f"Error fetching attendance logs: {exc}")
        return []


def archive_old_attendance():
    """Move attendance records older than the session window to the archive table."""
    try:
        with _db_cursor(commit=True) as cur:
            cutoff = datetime.now() - timedelta(minutes=config.SESSION_DURATION_MINUTES)
            cur.execute(
                "INSERT INTO attendance_archive (id, member_id, check_in_time) "
                "SELECT id, member_id, check_in_time "
                "FROM attendance_log "
                "WHERE check_in_time <= %s",
                (cutoff,),
            )
            archived = cur.rowcount
            if archived:
                cur.execute(
                    "DELETE FROM attendance_log WHERE check_in_time <= %s",
                    (cutoff,),
                )
                print(f"Archived {archived} attendance record(s).")
            return archived
    except Exception as exc:
        print(f"Error archiving attendance records: {exc}")
        return 0


def _delete_snapshot_files(image_paths):
    """Remove snapshot image files from disk, ignoring missing files."""
    for path in image_paths:
        try:
            if path and os.path.isfile(path):
                os.unlink(path)
        except OSError:
            pass


def _get_group_image_paths(cur, group_id):
    """Fetch all image_path values for a given unknown detection group."""
    cur.execute(
        "SELECT image_path FROM unknown_detections WHERE group_id = %s",
        (group_id,),
    )
    return [row[0] for row in cur.fetchall()]


def _parse_embedding(value):
    """Parse an embedding from pgvector, which may be a string or a list."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def enroll_from_unknown(group_id, name):
    """Enroll a member from unknown detection group. Returns member_id or None."""
    try:
        with _db_cursor(commit=True) as cur:
            cur.execute(
                "SELECT face_embedding FROM unknown_detections WHERE group_id = %s",
                (group_id,),
            )
            rows = cur.fetchall()
            if not rows:
                return None

            embeddings = [np.array(_parse_embedding(row[0])) for row in rows]
            avg_embedding = np.mean(embeddings, axis=0).tolist()

            cur.execute(
                "INSERT INTO members (name, face_embedding) VALUES (%s, %s) RETURNING id",
                (name, avg_embedding),
            )
            member_id = cur.fetchone()[0]

            image_paths = _get_group_image_paths(cur, group_id)

            cur.execute(
                "DELETE FROM unknown_detections WHERE group_id = %s",
                (group_id,),
            )

        _delete_snapshot_files(image_paths)
        return member_id
    except Exception as e:
        print(f"Error enrolling from unknown: {e}")
        return None


def dismiss_unknown_group(group_id):
    """Delete an unknown group and its snapshots. Returns True on success."""
    try:
        with _db_cursor(commit=True) as cur:
            image_paths = _get_group_image_paths(cur, group_id)

            cur.execute(
                "DELETE FROM unknown_detections WHERE group_id = %s",
                (group_id,),
            )

        _delete_snapshot_files(image_paths)
        return True
    except Exception as e:
        print(f"Error dismissing unknown group: {e}")
        return False
