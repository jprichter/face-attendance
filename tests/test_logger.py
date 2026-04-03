import pytest
from datetime import datetime

from logger import (
    log_check_in,
    log_unknown_detection,
    find_matching_unknown_group,
    get_unknown_groups,
    enroll_from_unknown,
    dismiss_unknown_group,
    save_member_image,
)


def test_log_check_in_cooldown(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn
    mock_cur.fetchone.return_value = (1,)

    result = log_check_in(101)

    assert result is False
    mock_cur.execute.assert_called()
    assert mock_cur.execute.call_args[0][1][0] == 101


def test_log_check_in_success(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn
    mock_cur.fetchone.return_value = None

    result = log_check_in(101)

    assert result is True
    assert any("INSERT INTO attendance_log" in call[0][0] for call in mock_cur.execute.call_args_list)


def test_log_unknown_detection(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn

    embedding = [0.1] * 512
    image_path = "test.jpg"

    result = log_unknown_detection(embedding, image_path)

    assert result is True
    assert any("INSERT INTO unknown_detections" in call[0][0] for call in mock_cur.execute.call_args_list)
    assert mock_cur.execute.call_args_list[-1][0][1] == (embedding, image_path, None)


def test_find_matching_unknown_group_match(mock_db_conn, mocker):
    _mock_connect, mock_cur = mock_db_conn
    embedding = [0.1] * 512

    mock_cur.fetchone.return_value = ("group-uuid-1", 0.3)
    mocker.patch("logger.config.RECOGNITION_THRESHOLD", 0.5)

    result = find_matching_unknown_group(embedding)
    assert result == "group-uuid-1"


def test_find_matching_unknown_group_no_match(mock_db_conn, mocker):
    _mock_connect, mock_cur = mock_db_conn
    embedding = [0.1] * 512

    mock_cur.fetchone.return_value = ("group-uuid-1", 0.8)
    mocker.patch("logger.config.RECOGNITION_THRESHOLD", 0.5)

    result = find_matching_unknown_group(embedding)
    assert result is None


def test_find_matching_unknown_group_empty_table(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn
    embedding = [0.1] * 512

    mock_cur.fetchone.return_value = None

    result = find_matching_unknown_group(embedding)
    assert result is None


def test_get_unknown_groups(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn

    expected = [
        ("group-1", "data/unknown/img1.jpg", 3, datetime(2023, 10, 27, 10, 0), datetime(2023, 10, 27, 10, 15)),
    ]
    mock_cur.fetchall.return_value = expected

    result = get_unknown_groups()
    assert result == expected


def test_get_unknown_groups_empty(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn
    mock_cur.fetchall.return_value = []

    result = get_unknown_groups()
    assert result == []


def test_enroll_from_unknown_success(mock_db_conn, mocker):
    _mock_connect, mock_cur = mock_db_conn
    mocker.patch("os.path.isfile", return_value=True)
    mocker.patch("os.unlink")

    mock_cur.fetchall.side_effect = [
        [([0.1] * 512,), ([0.2] * 512,)],  # embeddings
        [("data/unknown/img1.jpg",)],         # image paths
    ]
    mock_cur.fetchone.return_value = (42,)    # member_id

    result = enroll_from_unknown("group-uuid-1", "John Doe")
    assert result == 42


def test_enroll_from_unknown_no_detections(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn

    mock_cur.fetchall.return_value = []

    result = enroll_from_unknown("nonexistent", "John Doe")
    assert result is None


def test_dismiss_unknown_group_success(mock_db_conn, mocker):
    _mock_connect, mock_cur = mock_db_conn
    mocker.patch("os.path.isfile", return_value=True)
    mocker.patch("os.unlink")

    mock_cur.fetchall.return_value = [("data/unknown/img1.jpg",)]

    result = dismiss_unknown_group("group-uuid-1")
    assert result is True


def test_dismiss_unknown_group_db_error(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn

    mock_cur.execute.side_effect = Exception("DB error")

    result = dismiss_unknown_group("group-uuid-1")
    assert result is False


def test_save_member_image_success(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn
    mock_cur.rowcount = 1

    result = save_member_image(101, "data/members/member_101.jpg")

    assert result is True
    assert any("UPDATE members SET image_path" in str(call) for call in mock_cur.execute.call_args_list)


def test_save_member_image_already_exists(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn
    mock_cur.rowcount = 0

    result = save_member_image(101, "data/members/member_101.jpg")

    assert result is False
