import numpy as np
import pytest

from monitor import main as monitor_main, handle_recognition


@pytest.fixture(autouse=True)
def mock_monitor_infra(mocker):
    """Mock blocking infrastructure in monitor.py for all tests in this module."""
    mocker.patch("builtins.input", return_value="n")
    mocker.patch("monitor.threading.Thread")
    return


def test_full_flow_success(mock_db_conn, mock_deepface, mock_deepface_extract, mock_yunet, mocker):
    _mock_connect, mock_cur = mock_db_conn

    mock_cap = mocker.patch("cv2.VideoCapture")
    mock_instance = mock_cap.return_value
    mock_instance.isOpened.return_value = True

    dummy_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    mock_instance.read.side_effect = [(True, dummy_frame), (False, None)]

    # YuNet returns array with shape (N, 15): x, y, w, h, landmarks..., score
    face_detection = np.array([[100, 100, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]], dtype=np.float32)
    mock_yunet.detect.return_value = (1, face_detection)

    mock_deepface_extract.return_value = [{
        'face': np.zeros((160, 160, 3), dtype=np.float32),
        'facial_area': {'x': 0, 'y': 0, 'w': 10, 'h': 10},
        'confidence': 0.9,
    }]
    mock_deepface.return_value = [{"embedding": [0.1] * 512}]

    # First: query_database match; Second: log_check_in cooldown check (none found)
    mock_cur.fetchone.side_effect = [(101, "Test User", 0.1), None]

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.imwrite")
    mocker.patch("cv2.waitKey", return_value=ord('q'))
    mocker.patch("cv2.destroyAllWindows")
    mocker.patch("config.CONFIRMATION_FRAMES", 1)

    monitor_main()

    assert any("INSERT INTO attendance_log" in str(call) for call in mock_cur.execute.call_args_list)


def test_no_faces_detected_skips_deepface(mock_db_conn, mock_deepface, mock_deepface_extract, mock_yunet, mocker):
    """When YuNet detects no faces, DeepFace should not be called at all."""
    _mock_connect, _mock_cur = mock_db_conn

    mock_cap = mocker.patch("cv2.VideoCapture")
    mock_instance = mock_cap.return_value
    mock_instance.isOpened.return_value = True

    dummy_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    mock_instance.read.side_effect = [(True, dummy_frame), (False, None)]

    mock_yunet.detect.return_value = (0, None)

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=ord('q'))
    mocker.patch("cv2.destroyAllWindows")

    monitor_main()

    mock_deepface_extract.assert_not_called()
    mock_deepface.assert_not_called()


def test_single_stage_low_res(mock_db_conn, mock_deepface, mock_deepface_extract, mocker):
    """Low-res webcam frames should use single-stage pipeline (no YuNet)."""
    _mock_connect, mock_cur = mock_db_conn

    mock_cap = mocker.patch("cv2.VideoCapture")
    mock_instance = mock_cap.return_value
    mock_instance.isOpened.return_value = True

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_instance.read.side_effect = [(True, dummy_frame), (False, None)]

    mock_deepface_extract.return_value = [{
        'face': np.zeros((160, 160, 3), dtype=np.float32),
        'facial_area': {'x': 0, 'y': 0, 'w': 10, 'h': 10},
        'confidence': 0.9,
    }]
    mock_deepface.return_value = [{"embedding": [0.1] * 512}]

    mock_cur.fetchone.side_effect = [(101, "Test User", 0.1), None]

    mocker.patch("cv2.imshow")
    mocker.patch("cv2.imwrite")
    mocker.patch("cv2.waitKey", return_value=ord('q'))
    mocker.patch("cv2.destroyAllWindows")
    mocker.patch("config.CONFIRMATION_FRAMES", 1)

    mock_yunet_create = mocker.patch("cv2.FaceDetectorYN.create")
    mocker.patch("monitor.ensure_yunet_model")

    monitor_main()

    mock_yunet_create.assert_not_called()
    assert any("INSERT INTO attendance_log" in str(call) for call in mock_cur.execute.call_args_list)


def test_unknown_face_grouping(mock_db_conn, mocker):
    """Test that unknown faces get assigned a group_id and logged."""
    mocker.patch("monitor.query_database", return_value=(None, "Unknown", None))
    mock_find = mocker.patch("monitor.find_matching_unknown_group", return_value=None)
    mock_log = mocker.patch("monitor.log_unknown_detection")
    mocker.patch("cv2.imwrite")
    mocker.patch("os.path.join", return_value="data/unknown/unknown_test.jpg")

    embedding = np.random.rand(512).tolist()
    confirmation_buffer = {"id": None, "count": 0}
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)

    name = handle_recognition(embedding, confirmation_buffer, fake_image)

    assert name == "Unknown"
    mock_find.assert_called_once_with(embedding)
    mock_log.assert_called_once()
    call_args = mock_log.call_args[0]
    assert call_args[0] == embedding
    assert call_args[1] == "data/unknown/unknown_test.jpg"
    assert call_args[2] is not None
    assert len(call_args[2]) == 36  # UUID format


def test_unknown_face_existing_group(mock_db_conn, mocker):
    """Test that unknown faces matching an existing group reuse the group_id."""
    mocker.patch("monitor.query_database", return_value=(None, "Unknown", 0.7))
    mocker.patch("monitor.find_matching_unknown_group", return_value="existing-group-uuid")
    mock_log = mocker.patch("monitor.log_unknown_detection")
    mocker.patch("cv2.imwrite")
    mocker.patch("os.path.join", return_value="data/unknown/unknown_test.jpg")

    embedding = np.random.rand(512).tolist()
    confirmation_buffer = {"id": None, "count": 0}
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)

    name = handle_recognition(embedding, confirmation_buffer, fake_image)

    assert name == "Unknown"
    assert mock_log.call_args[0][2] == "existing-group-uuid"
