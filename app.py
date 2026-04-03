from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory

import config
from logger import (
    get_attendance_logs,
    get_unknown_groups,
    enroll_from_unknown,
    dismiss_unknown_group,
)

app = Flask(__name__)


@app.route('/')
def index():
    logs = get_attendance_logs()
    unknown_groups = get_unknown_groups()
    return render_template('index.html', logs=logs, unknown_groups=unknown_groups, now=datetime.now())


@app.route('/snapshots/<filename>')
def serve_snapshot(filename):
    return send_from_directory(config.UNKNOWN_FACES_DIR, filename)


@app.route('/member-photos/<filename>')
def serve_member_photo(filename):
    return send_from_directory(config.MEMBER_FACES_DIR, filename)


@app.route('/enroll', methods=['POST'])
def enroll():
    data = request.get_json()
    if not data or not data.get('group_id') or not data.get('name', '').strip():
        return jsonify({"error": "Missing name or group_id"}), 400

    member_id = enroll_from_unknown(data['group_id'], data['name'].strip())
    if member_id is None:
        return jsonify({"error": "No detections found for group_id"}), 404

    return jsonify({"success": True, "member_id": member_id, "name": data['name'].strip()})


@app.route('/dismiss', methods=['POST'])
def dismiss():
    data = request.get_json()
    if not data or not data.get('group_id'):
        return jsonify({"error": "Missing group_id"}), 400

    success = dismiss_unknown_group(data['group_id'])
    if not success:
        return jsonify({"error": "Failed to dismiss group"}), 500

    return jsonify({"success": True})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=True)
