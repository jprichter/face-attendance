import pytest
from datetime import datetime, timedelta
from logger import log_check_in, log_unknown_detection
import config

def test_log_check_in_cooldown(mock_db_conn):
    # Get mock connection and cursor from fixture
    mock_conn, mock_cur = mock_db_conn
    
    # Simulate that an entry WAS found within the cooldown period
    mock_cur.fetchone.return_value = (1,) # Found an existing ID
    
    result = log_check_in(101)
    
    assert result is False
    # Verify the query was executed with the member_id
    mock_cur.execute.assert_called()
    assert mock_cur.execute.call_args[0][1][0] == 101

def test_log_check_in_success(mock_db_conn):
    mock_conn, mock_cur = mock_db_conn
    
    # Simulate NO existing entry found
    mock_cur.fetchone.return_value = None
    
    result = log_check_in(101)
    
    assert result is True
    # Verify INSERT was called
    assert any("INSERT INTO attendance_log" in call[0][0] for call in mock_cur.execute.call_args_list)

def test_log_unknown_detection(mock_db_conn):
    mock_conn, mock_cur = mock_db_conn
    
    embedding = [0.1] * 512
    image_path = "test.jpg"
    
    result = log_unknown_detection(embedding, image_path)
    
    assert result is True
    # Verify INSERT was called with correct data
    assert any("INSERT INTO unknown_detections" in call[0][0] for call in mock_cur.execute.call_args_list)
    assert mock_cur.execute.call_args_list[-1][0][1] == (embedding, image_path)
