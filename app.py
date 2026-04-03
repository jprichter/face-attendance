import threading
import time
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory

import config
from logger import (
    archive_old_attendance,
    dismiss_unknown_group,
    enroll_from_unknown,
    ensure_archive_table,
    get_attendance_logs,
    get_member_names,
    get_unknown_groups,
)

app = Flask(__name__)


def _run_archive_loop():
    """Periodically archive expired attendance records."""
    while True:
        time.sleep(config.ARCHIVE_INTERVAL_MINUTES * 60)
        archive_old_attendance()


ensure_archive_table()
threading.Thread(target=_run_archive_loop, daemon=True).start()


@app.route('/')
def index():
    logs = get_attendance_logs()
    member_names = get_member_names()
    unknown_groups = get_unknown_groups()
    return render_template(
        'index.html',
        logs=logs,
        member_names=member_names,
        unknown_groups=unknown_groups,
        now=datetime.now(),
    )


@app.route('/snapshots/<filename>')
def serve_snapshot(filename):
    return send_from_directory(config.UNKNOWN_FACES_DIR, filename)


@app.route('/member-photos/<filename>')
def serve_member_photo(filename):
    return send_from_directory(config.MEMBER_FACES_DIR, filename)


@app.route('/enroll', methods=['POST'])
def enroll():
    data = request.get_json()
    detection_ids = data.get('detection_ids') if data else None
    if not data or not data.get('group_id') or not data.get('name', '').strip():
        return jsonify({"error": "Missing name or group_id"}), 400
    if not isinstance(detection_ids, list) or not detection_ids:
        return jsonify({"error": "Select at least one photo"}), 400

    result = enroll_from_unknown(data['group_id'], data['name'].strip(), detection_ids)
    if result is None:
        return jsonify({"error": "No selected detections found for group_id"}), 404

    return jsonify({
        "success": True,
        "member_id": result["member_id"],
        "name": data['name'].strip(),
        "action": result["action"],
    })


@app.route('/dismiss', methods=['POST'])
def dismiss():
    data = request.get_json()
    detection_ids = data.get('detection_ids') if data else None
    if not data or not data.get('group_id'):
        return jsonify({"error": "Missing group_id"}), 400
    if not isinstance(detection_ids, list) or not detection_ids:
        return jsonify({"error": "Select at least one photo"}), 400

    success = dismiss_unknown_group(data['group_id'], detection_ids)
    if not success:
        return jsonify({"error": "Failed to dismiss selected photos"}), 500

    return jsonify({"success": True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=True)
