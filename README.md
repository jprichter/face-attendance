# Face Attendance PoC 🛡️📸

A real-time automated attendance tracking system using facial recognition. Built with DeepFace, PostgreSQL (pgvector), and Flask.

## Features
- **Real-time Recognition**: identify enrolled members via webcam or RTSP stream.
- **pgvector Integration**: Efficient vector similarity search for face embeddings.
- **Robust Enrollment**: Pre-checks DB connections and caches embeddings locally if the database is offline.
- **Smart Logging**: 3-frame confirmation logic to prevent false positives and configurable cool-down periods.
- **Web Dashboard**: Simple Flask interface to view attendance logs in real-time.
- **Unknown Face Capture**: Automatically logs and saves snapshots of unauthorized/unknown individuals.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- **Ubuntu Linux** or **macOS**.
- **Python 3.11+**.
- **PostgreSQL 12+**.
- **uv**: Modern Python package and project manager. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Database Setup
```bash
# Install Postgres, pgvector, and setup the schema
./bin/setup_postgres.sh

# Fix Database Authentication (if you get password errors)
./bin/fix_db_auth.sh
```

### 3. Setup Python Virtual Environment and Dependencies
```bash
uv sync
```

### 4. Configuration
Copy the example environment file and adjust your settings (DB credentials, camera source, etc.):
```bash
cp .env.example .env
# Edit .env with your favorite editor
```

---

## 🚀 How to Use

### Step 1: Enroll Members
Organize photos in `data/faces/NAME/` and run:
```bash
uv run enroll.py --name "Your Name" --folder data/faces/your_name/
```

### Step 2: Run Monitoring
Start the recognition loop:
```bash
uv run monitor.py
```
*Note: The terminal will provide a link to the Web Log Viewer (usually http://localhost:5000).*

### Step 3: View Logs
Open your browser to the link provided by `monitor.py` to see the real-time check-in table.

---

## 🧪 Running Tests
The project uses `pytest` for its test suite. Configuration is handled in `pytest.ini`, and shared fixtures (like database and AI mocks) are located in `tests/conftest.py`.

To run the entire suite:
```bash
uv run pytest
```

To run specific test categories:
- **Unit Tests**: `uv run pytest tests/test_logger.py tests/test_enroll.py`
- **Web Tests**: `uv run pytest tests/test_app.py`
- **E2E Tests**: `uv run pytest tests/test_e2e.py`

Test results are saved in the `test_results/` directory using the format `{name_of_test}-{timestamp}.txt`.

---

## 📂 Project Structure
- `enroll.py`: Script to add new faces to the DB.
- `monitor.py`: The core recognition engine.
- `app.py`: Flask web server for the log viewer.
- `logger.py`: Database logic for check-ins and unknown faces.
- `config.py`: Centralized configuration management.
- `bin/`: Setup and utility bash scripts.
- `docs/`: Design plans, database schema, and user manuals.

---

## 👥 Contributors
- jprichter
- sdiviney
