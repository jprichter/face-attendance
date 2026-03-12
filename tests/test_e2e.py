import pytest
import cv2
import numpy as np
import config
from monitor import main as monitor_main

def test_full_flow_success(mock_db_conn, mock_deepface, mock_deepface_extract, mocker):
    # Get mock connection and cursor from fixture
    mock_conn, mock_cur = mock_db_conn
    
    # Mock camera to return one frame then exit
    mock_cap = mocker.patch("cv2.VideoCapture")
    mock_instance = mock_cap.return_value
    mock_instance.isOpened.return_value = True
    
    # Return a dummy frame once, then False to exit loop
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_instance.read.side_effect = [(True, dummy_frame), (False, None)]
    
    # Mock DeepFace.extract_faces to find one face
    mock_deepface_extract.return_value = [{
        'face': dummy_frame,
        'facial_area': {'x': 0, 'y': 0, 'w': 10, 'h': 10},
        'confidence': 0.9
    }]
    
    # Mock DeepFace.represent to return dummy embedding
    mock_deepface.return_value = [{"embedding": [0.1] * 512}]
    
    # First query (query_database): Return a match
    # Second query (log_check_in -> cooldown check): Return None (no cooldown)
    mock_cur.fetchone.side_effect = [(101, "Test User", 0.1), None]
    
    # Mock cv2.imshow to avoid opening windows during test
    mocker.patch("cv2.imshow")
    mocker.patch("cv2.waitKey", return_value=ord('q')) # Force exit
    
    # We need to mock CONFIRMATION_FRAMES to 1 for this test to trigger log_check_in quickly
    mocker.patch("config.CONFIRMATION_FRAMES", 1)
    
    # Run the monitor main loop
    monitor_main()
    
    # Assertions
    # Verify log_check_in was reached (via DB call)
    assert any("INSERT INTO attendance_log" in str(call) for call in mock_cur.execute.call_args_list)
