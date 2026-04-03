import pytest
import cv2
import numpy as np
import config
from monitor import main as monitor_main


@pytest.fixture(autouse=True)
def mock_monitor_infra(mocker):
    """Mock blocking infrastructure in monitor.py for all tests in this module."""
    mocker.patch("builtins.input", return_value="n")
    mocker.patch("monitor.threading.Thread")
    return


def test_full_flow_success(mock_db_conn, mock_deepface, mock_deepface_extract, mock_yunet, mocker):
    # Get mock connection and cursor from fixture
    mock_conn, mock_cur = mock_db_conn

    # Mock camera to return one frame then exit
    mock_cap = mocker.patch("cv2.VideoCapture")
    mock_instance = mock_cap.return_value
    mock_instance.isOpened.return_value = True

    # Return a dummy 4K frame once, then False to exit loop
    dummy_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    mock_instance.read.side_effect = [(True, dummy_frame), (False, None)]

    # Mock YuNet to detect one face at (100, 100, 50, 50) in detection frame coordinates
    # YuNet returns array with shape (N, 15): x, y, w, h, landmarks..., score
    face_detection = np.array([[100, 100, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]], dtype=np.float32)
    mock_yunet.detect.return_value = (1, face_detection)

    # Mock DeepFace.extract_faces to find one face in the crop
    mock_deepface_extract.return_value = [{
        'face': np.zeros((160, 160, 3), dtype=np.float32),
        'facial_area': {'x': 0, 'y': 0, 'w': 10, 'h': 10},
        'confidence': 0.9
    }]

    # Mock DeepFace.represent to return dummy embedding
    mock_deepface.return_value = [{"embedding": [0.1] * 512}]

    # First query (query_database): Return a match
    # Second query (log_check_in -> cooldown check): Return None (no cooldown)
    mock_cur.fetchone.side_effect = [(101, "Test User", 0.1), None]

    # Mock cv2.imshow and resize to avoid GUI operations
    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=ord('q'))
    mocker.patch("cv2.destroyAllWindows")

    # Set CONFIRMATION_FRAMES to 1 so log_check_in triggers on first match
    mocker.patch("config.CONFIRMATION_FRAMES", 1)

    # Run the monitor main loop
    monitor_main()

    # Verify log_check_in was reached (via DB call)
    assert any("INSERT INTO attendance_log" in str(call) for call in mock_cur.execute.call_args_list)


def test_no_faces_detected_skips_deepface(mock_db_conn, mock_deepface, mock_deepface_extract, mock_yunet, mocker):
    """When YuNet detects no faces, DeepFace should not be called at all."""
    mock_conn, mock_cur = mock_db_conn

    mock_cap = mocker.patch("cv2.VideoCapture")
    mock_instance = mock_cap.return_value
    mock_instance.isOpened.return_value = True

    dummy_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    mock_instance.read.side_effect = [(True, dummy_frame), (False, None)]

    # YuNet detects no faces
    mock_yunet.detect.return_value = (0, None)

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=ord('q'))
    mocker.patch("cv2.destroyAllWindows")

    monitor_main()

    # DeepFace should never have been called
    mock_deepface_extract.assert_not_called()
    mock_deepface.assert_not_called()


def test_single_stage_low_res(mock_db_conn, mock_deepface, mock_deepface_extract, mocker):
    """Low-res webcam frames should use single-stage pipeline (no YuNet)."""
    mock_conn, mock_cur = mock_db_conn

    mock_cap = mocker.patch("cv2.VideoCapture")
    mock_instance = mock_cap.return_value
    mock_instance.isOpened.return_value = True

    # 640x480 webcam frame — at or below DETECTION_WIDTH, triggers single-stage
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_instance.read.side_effect = [(True, dummy_frame), (False, None)]

    # Mock DeepFace.extract_faces to find one face
    mock_deepface_extract.return_value = [{
        'face': np.zeros((160, 160, 3), dtype=np.float32),
        'facial_area': {'x': 0, 'y': 0, 'w': 10, 'h': 10},
        'confidence': 0.9
    }]

    # Mock DeepFace.represent to return dummy embedding
    mock_deepface.return_value = [{"embedding": [0.1] * 512}]

    # DB returns a match, then no cooldown
    mock_cur.fetchone.side_effect = [(101, "Test User", 0.1), None]

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=ord('q'))
    mocker.patch("cv2.destroyAllWindows")
    mocker.patch("config.CONFIRMATION_FRAMES", 1)

    # YuNet should NOT be initialized — mock to verify it's never called
    mock_yunet_create = mocker.patch("cv2.FaceDetectorYN.create")
    mocker.patch("monitor.ensure_yunet_model")

    monitor_main()

    # YuNet should not have been used
    mock_yunet_create.assert_not_called()
    # But DeepFace should have processed the frame directly
    assert any("INSERT INTO attendance_log" in str(call) for call in mock_cur.execute.call_args_list)
