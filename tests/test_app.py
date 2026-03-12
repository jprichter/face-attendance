import pytest
from app import app
from datetime import datetime

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client, mock_db_conn):
    # Get mock connection and cursor from fixture
    mock_conn, mock_cur = mock_db_conn
    
    # Mock some attendance logs
    mock_cur.fetchall.return_value = [
        (1, "John Doe", datetime(2023, 10, 27, 10, 0, 0)),
        (2, "Jane Smith", datetime(2023, 10, 27, 10, 5, 0))
    ]
    
    response = client.get('/')
    assert response.status_code == 200
    
    # Check if the content contains the names from our mock
    assert b"John Doe" in response.data
    assert b"Jane Smith" in response.data
    assert b"Attendance Logs" in response.data

def test_index_no_logs(client, mock_db_conn):
    # Get mock connection and cursor from fixture
    mock_conn, mock_cur = mock_db_conn
    mock_cur.fetchall.return_value = []
    
    response = client.get('/')
    assert response.status_code == 200
    assert b"No logs found" in response.data
