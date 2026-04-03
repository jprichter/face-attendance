import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_db_conn(mocker):
    """Common fixture to mock psycopg2 database connection."""
    mock_conn = mocker.patch("psycopg2.connect")
    mock_cur = mock_conn.return_value.cursor.return_value
    return mock_conn, mock_cur

@pytest.fixture
def mock_deepface(mocker):
    """Common fixture to mock DeepFace.represent."""
    mock_df = mocker.patch("deepface.DeepFace.represent")
    return mock_df

@pytest.fixture
def mock_deepface_extract(mocker):
    """Common fixture to mock DeepFace.extract_faces."""
    mock_df_ext = mocker.patch("deepface.DeepFace.extract_faces")
    return mock_df_ext

@pytest.fixture
def mock_yunet(mocker):
    """Mock YuNet detector for two-stage pipeline."""
    mocker.patch("monitor.ensure_yunet_model")
    mock_detector = MagicMock()
    mocker.patch("cv2.FaceDetectorYN.create", return_value=mock_detector)
    return mock_detector
