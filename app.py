from flask import Flask, render_template
import psycopg2
import config

app = Flask(__name__)

def get_db_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        database=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASS,
        port=config.DB_PORT
    )

@app.route('/')
def index():
    conn = None
    logs = []
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Fetch logs with member names
        cur.execute(
            "SELECT a.id, m.name, a.check_in_time "
            "FROM attendance_log a "
            "JOIN members m ON a.member_id = m.id "
            "ORDER BY a.check_in_time DESC LIMIT 100"
        )
        logs = cur.fetchall()
    except Exception as e:
        print(f"Web error: {e}")
    finally:
        if conn:
            conn.close()
    
    from datetime import datetime
    return render_template('index.html', logs=logs, now=datetime.now())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=True)
