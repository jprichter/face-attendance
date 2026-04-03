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
    """Get unknown face groups with their selectable detections for the dashboard."""
    try:
        with _db_cursor() as cur:
            cur.execute(
                "SELECT ud.id, ud.group_id, ud.image_path, ud.detected_at "
                "FROM unknown_detections ud "
                "WHERE ud.group_id IS NOT NULL "
                "ORDER BY ud.group_id, ud.detected_at DESC"
            )
            rows = cur.fetchall()

        groups = {}
        for detection_id, group_id, image_path, detected_at in rows:
            group = groups.setdefault(group_id, {
                "group_id": group_id,
                "seen_count": 0,
                "first_seen": detected_at,
                "last_seen": detected_at,
                "detections": [],
            })
            group["seen_count"] += 1
            if detected_at < group["first_seen"]:
                group["first_seen"] = detected_at
            if detected_at > group["last_seen"]:
                group["last_seen"] = detected_at
            group["detections"].append({
                "id": detection_id,
                "image_path": image_path,
                "detected_at": detected_at,
            })

        return sorted(groups.values(), key=lambda group: group["last_seen"], reverse=True)
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


def get_member_names():
    """Fetch member names for enrollment suggestions."""
    try:
        with _db_cursor() as cur:
            cur.execute(
                "SELECT id, name "
                "FROM members "
                "ORDER BY LOWER(name) ASC"
            )
            return cur.fetchall()
    except Exception as e:
        print(f"Error fetching member names: {e}")
        return []


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


def _normalize_detection_ids(detection_ids):
    if not detection_ids:
        return []
    normalized = []
    for detection_id in detection_ids:
        try:
            normalized.append(int(detection_id))
        except (TypeError, ValueError):
            continue
    return sorted(set(normalized))


def _get_selected_detections(cur, group_id, detection_ids):
    normalized_ids = _normalize_detection_ids(detection_ids)
    if not normalized_ids:
        return []

    cur.execute(
        "SELECT id, face_embedding, image_path "
        "FROM unknown_detections "
        "WHERE group_id = %s AND id = ANY(%s)",
        (group_id, normalized_ids),
    )
    return cur.fetchall()


def _parse_embedding(value):
    """Parse an embedding from pgvector, which may be a string or a list."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _find_member_by_name(cur, name):
    cur.execute(
        "SELECT id, name, face_embedding "
        "FROM members "
        "WHERE LOWER(TRIM(name)) = LOWER(TRIM(%s)) "
        "LIMIT 1",
        (name,),
    )
    return cur.fetchone()


def enroll_from_unknown(group_id, name, detection_ids):
    """Create or update a member from selected detections within an unknown group."""
    try:
        with _db_cursor(commit=True) as cur:
            rows = _get_selected_detections(cur, group_id, detection_ids)
            if not rows:
                return None

            selected_ids = [row[0] for row in rows]
            embeddings = [np.array(_parse_embedding(row[1])) for row in rows]
            selected_embedding = np.mean(embeddings, axis=0)
            image_paths = [row[2] for row in rows]

            existing_member = _find_member_by_name(cur, name)
            if existing_member:
                member_id = existing_member[0]
                existing_embedding = np.array(_parse_embedding(existing_member[2]))
                merged_embedding = np.mean([existing_embedding, selected_embedding], axis=0).tolist()
                cur.execute(
                    "UPDATE members SET face_embedding = %s WHERE id = %s",
                    (merged_embedding, member_id),
                )
                action = "updated"
            else:
                cur.execute(
                    "INSERT INTO members (name, face_embedding) VALUES (%s, %s) RETURNING id",
                    (name, selected_embedding.tolist()),
                )
                member_id = cur.fetchone()[0]
                action = "created"

            cur.execute(
                "DELETE FROM unknown_detections "
                "WHERE group_id = %s AND id = ANY(%s)",
                (group_id, selected_ids),
            )

        _delete_snapshot_files(image_paths)
        return {"member_id": member_id, "action": action}
    except Exception as e:
        print(f"Error enrolling from unknown: {e}")
        return None


def dismiss_unknown_group(group_id, detection_ids):
    """Delete selected detections from an unknown group. Returns True on success."""
    try:
        with _db_cursor(commit=True) as cur:
            rows = _get_selected_detections(cur, group_id, detection_ids)
            if not rows:
                return False

            selected_ids = [row[0] for row in rows]
            image_paths = [row[2] for row in rows]

            cur.execute(
                "DELETE FROM unknown_detections "
                "WHERE group_id = %s AND id = ANY(%s)",
                (group_id, selected_ids),
            )

        _delete_snapshot_files(image_paths)
        return True
    except Exception as e:
        print(f"Error dismissing unknown group: {e}")
        return False
