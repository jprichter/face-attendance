import pytest
import os
import numpy as np
from enroll import enroll_member
import config

def test_enroll_member_success(mock_db_conn, mock_deepface, mocker):
    # Get mock connection and cursor from fixture
    mock_conn, mock_cur = mock_db_conn
    
    # Mock DeepFace.represent to return a dummy embedding
    mock_deepface.return_value = [{"embedding": [0.1] * 512}]
    
    # Mock os.listdir to simulate finding images
    mocker.patch("os.listdir", return_value=["face1.jpg"])
    
    # Run the function
    enroll_member("Test User", "/dummy/path")
    
    # Assertions
    # Verify DeepFace was called
    mock_deepface.assert_called_once()
    # Verify INSERT was called
    mock_cur.execute.assert_called()
    assert any("INSERT INTO members" in str(call) for call in mock_cur.execute.call_args_list)
    # Check that name was passed correctly
    assert any("Test User" in str(call) for call in mock_cur.execute.call_args_list)

def test_enroll_member_no_faces(mock_db_conn, mock_deepface, mocker):
    # Get mock connection and cursor from fixture
    mock_conn, mock_cur = mock_db_conn

    # Mock DeepFace.represent to return no results
    mock_deepface.return_value = []
    
    mocker.patch("os.listdir", return_value=["face1.jpg"])
    
    # Run the function
    enroll_member("Test User", "/dummy/path")
    
    # Assertions
    # Verify DeepFace was called but NO insert occurred
    mock_deepface.assert_called_once()
    mock_cur.execute.assert_not_called()
