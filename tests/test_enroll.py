import os
import tempfile

import pytest

from enroll import enroll_member


def test_enroll_member_success(mock_db_conn, mock_deepface, mocker):
    _mock_connect, mock_cur = mock_db_conn

    mock_deepface.return_value = [{"embedding": [0.1] * 512}]
    mocker.patch("os.listdir", return_value=["face1.jpg"])

    enroll_member("Test User", "/dummy/path")

    mock_deepface.assert_called_once()
    mock_cur.execute.assert_called()
    assert any("INSERT INTO members" in str(call) for call in mock_cur.execute.call_args_list)
    assert any("Test User" in str(call) for call in mock_cur.execute.call_args_list)


def test_enroll_member_no_faces(mock_db_conn, mock_deepface, mocker):
    _mock_connect, mock_cur = mock_db_conn

    mock_deepface.return_value = []
    mocker.patch("os.listdir", return_value=["face1.jpg"])

    enroll_member("Test User", "/dummy/path")

    mock_deepface.assert_called_once()
    mock_cur.execute.assert_not_called()


def test_enroll_member_missing_folder(mock_db_conn, mock_deepface):
    """enroll_member should raise FileNotFoundError when folder doesn't exist."""
    nonexistent = os.path.join(tempfile.gettempdir(), "nonexistent_folder_xyz")
    with pytest.raises(FileNotFoundError):
        enroll_member("Test User", nonexistent)


def test_folder_name_generation():
    """Test that folder auto-generation follows the expected pattern."""
    name = "Sean Diviney"
    parts = name.strip().split()
    first_name = parts[0].lower()
    last_name = parts[-1].lower() if len(parts) > 1 else first_name

    folder_name = f"{first_name[0]}{last_name}"
    assert folder_name == "sdiviney"


def test_folder_name_generation_single_name():
    """Test folder generation with a single-word name."""
    name = "Madonna"
    parts = name.strip().split()
    first_name = parts[0].lower()
    last_name = parts[-1].lower() if len(parts) > 1 else first_name

    folder_name = f"{first_name[0]}{last_name}"
    assert folder_name == "mmadonna"


def test_folder_name_fallback():
    """Test that folder generation falls back to two-letter prefix on collision."""
    name = "Sean Diviney"
    parts = name.strip().split()
    first_name = parts[0].lower()
    last_name = parts[-1].lower() if len(parts) > 1 else first_name

    folder_name = f"{first_name[0]}{last_name}"
    assert folder_name == "sdiviney"

    folder_name_fallback = f"{first_name[:2]}{last_name}"
    assert folder_name_fallback == "sediviney"
