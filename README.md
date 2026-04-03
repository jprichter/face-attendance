# Face Attendance PoC 🛡️📸

A real-time automated attendance tracking system using facial recognition. Built with DeepFace, PostgreSQL (pgvector), and Flask.

## Features
- **Real-time Recognition**: Identify enrolled members via webcam or RTSP stream.
- **pgvector Integration**: Efficient vector similarity search for face embeddings.
- **Unknown Face Grouping**: Automatically groups unknown faces by embedding similarity so repeat visitors are linked together.
- **Web Enrollment**: Review unknown face groups in the dashboard and enroll them with one click — no CLI needed.
- **CLI Enrollment**: Batch-enroll from a folder of photos. `--folder` is optional and auto-generates from `--name`.
- **Tabbed Dashboard**: Flask interface with an Attendance tab (check-in log with member photo thumbnails) and an Enroll tab (unknown face review queue).
- **Member Photo Thumbnails**: Captures a face snapshot on first check-in and displays it in the attendance table.
- **Smart Logging**: 3-frame confirmation logic to prevent false positives and configurable cool-down periods.
- **Offline Fallback**: Caches enrollment embeddings locally if the database is unreachable.

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

### Option A: Web Enrollment (recommended)

1. **Start the monitor** — detects faces and groups unknown visitors by similarity:
   ```bash
   uv run monitor.py
   ```

2. **Start the dashboard** (in a separate terminal):
   ```bash
   uv run app.py
   ```

3. **Enroll via the browser** — open http://localhost:5000, click the **Enroll** tab, enter a name for each unknown face group, and click Enroll.

### Option B: CLI Enrollment

```bash
# Auto-generate folder from name (creates data/faces/sdiviney/)
uv run enroll.py --name "Sean Diviney"

# Or specify an existing folder of photos
uv run enroll.py --name "Sean Diviney" --folder data/faces/sdiviney/
```

Then start the monitor and dashboard:
```bash
uv run monitor.py
uv run app.py     # http://localhost:5000
```

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
- `monitor.py`: Core recognition loop — face detection, embedding extraction, pgvector similarity search, unknown face grouping, member photo capture.
- `enroll.py`: CLI enrollment — processes photos, averages embeddings, stores in DB. Auto-generates folder from name.
- `app.py`: Flask dashboard — tabbed UI (Attendance + Enroll), serves member photos and unknown face snapshots.
- `logger.py`: DB operations — check-in logging, unknown detection with group linking, web enrollment, member photo storage.
- `config.py`: Centralized configuration from `.env`.
- `bin/`: Setup and utility bash scripts.
- `docs/`: Design plans, database schema, and user manuals.

---

## 👥 Contributors
- jprichter
- sdiviney
