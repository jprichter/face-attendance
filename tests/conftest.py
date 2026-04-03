import pytest


@pytest.fixture
def mock_db_conn(mocker):
    """Mock psycopg2.connect, returning (mock_connect, mock_cursor)."""
    mock_connect = mocker.patch("psycopg2.connect")
    mock_cursor = mock_connect.return_value.cursor.return_value
    return mock_connect, mock_cursor


@pytest.fixture
def mock_deepface(mocker):
    """Mock DeepFace.represent."""
    return mocker.patch("deepface.DeepFace.represent")


@pytest.fixture
def mock_deepface_extract(mocker):
    """Mock DeepFace.extract_faces."""
    return mocker.patch("deepface.DeepFace.extract_faces")


@pytest.fixture
def mock_yunet(mocker):
    """Mock YuNet detector for two-stage pipeline."""
    mocker.patch("monitor.ensure_yunet_model")
    mock_detector = mocker.MagicMock()
    mocker.patch("cv2.FaceDetectorYN.create", return_value=mock_detector)
    return mock_detector
