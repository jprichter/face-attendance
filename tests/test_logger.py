from datetime import datetime

import pytest

from logger import (
    archive_old_attendance,
    dismiss_unknown_group,
    enroll_from_unknown,
    find_matching_unknown_group,
    get_attendance_logs,
    get_member_names,
    get_unknown_groups,
    log_check_in,
    log_unknown_detection,
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

    mock_cur.fetchall.return_value = [
        (1, "group-1", "data/unknown/img2.jpg", datetime(2023, 10, 27, 10, 15)),
        (2, "group-1", "data/unknown/img1.jpg", datetime(2023, 10, 27, 10, 0)),
    ]

    result = get_unknown_groups()
    assert len(result) == 1
    assert result[0]["group_id"] == "group-1"
    assert result[0]["seen_count"] == 2
    assert result[0]["first_seen"] == datetime(2023, 10, 27, 10, 0)
    assert result[0]["last_seen"] == datetime(2023, 10, 27, 10, 15)
    assert result[0]["detections"][0]["id"] == 1


def test_get_unknown_groups_empty(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn
    mock_cur.fetchall.return_value = []

    result = get_unknown_groups()
    assert result == []


def test_get_member_names(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn
    mock_cur.fetchall.return_value = [(2, "Jane Smith"), (1, "John Doe")]

    result = get_member_names()
    assert result == [(2, "Jane Smith"), (1, "John Doe")]


def test_enroll_from_unknown_success(mock_db_conn, mocker):
    _mock_connect, mock_cur = mock_db_conn
    mocker.patch("os.path.isfile", return_value=True)
    mocker.patch("os.unlink")

    mock_cur.fetchall.side_effect = [
        [
            (10, [0.1] * 512, "data/unknown/img1.jpg"),
            (11, [0.2] * 512, "data/unknown/img2.jpg"),
        ],
    ]
    mock_cur.fetchone.side_effect = [None, (42,)]

    result = enroll_from_unknown("group-uuid-1", "John Doe", [10, 11])
    assert result == {"member_id": 42, "action": "created"}


def test_enroll_from_unknown_updates_existing_member(mock_db_conn, mocker):
    _mock_connect, mock_cur = mock_db_conn
    mocker.patch("os.path.isfile", return_value=True)
    mocker.patch("os.unlink")

    mock_cur.fetchall.side_effect = [
        [
            (10, [0.1] * 512, "data/unknown/img1.jpg"),
            (11, [0.3] * 512, "data/unknown/img2.jpg"),
        ],
    ]
    mock_cur.fetchone.return_value = (7, "John Doe", [0.5] * 512)

    result = enroll_from_unknown("group-uuid-1", "John Doe", [10, 11])
    assert result == {"member_id": 7, "action": "updated"}
    assert any("UPDATE members SET face_embedding" in call[0][0] for call in mock_cur.execute.call_args_list)


def test_enroll_from_unknown_no_detections(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn

    mock_cur.fetchall.return_value = []

    result = enroll_from_unknown("nonexistent", "John Doe", [999])
    assert result is None


def test_dismiss_unknown_group_success(mock_db_conn, mocker):
    _mock_connect, mock_cur = mock_db_conn
    mocker.patch("os.path.isfile", return_value=True)
    mocker.patch("os.unlink")

    mock_cur.fetchall.return_value = [(10, [0.1] * 512, "data/unknown/img1.jpg")]

    result = dismiss_unknown_group("group-uuid-1", [10])
    assert result is True


def test_dismiss_unknown_group_db_error(mock_db_conn):
    _mock_connect, mock_cur = mock_db_conn

    mock_cur.execute.side_effect = Exception("DB error")

    result = dismiss_unknown_group("group-uuid-1", [10])
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


def test_get_attendance_logs_filters_by_session(mock_db_conn, mocker):
    """get_attendance_logs passes session window cutoff to the query."""
    mocker.patch("logger.config.SESSION_DURATION_MINUTES", 90)
    _mock_connect, mock_cur = mock_db_conn
    mock_cur.fetchall.return_value = []
    get_attendance_logs()
    sql = mock_cur.execute.call_args[0][0]
    assert "check_in_time >" in sql


def test_archive_old_attendance(mock_db_conn, mocker):
    """archive_old_attendance moves expired records to archive table."""
    mocker.patch("logger.config.SESSION_DURATION_MINUTES", 90)
    _mock_connect, mock_cur = mock_db_conn
    mock_cur.rowcount = 3
    result = archive_old_attendance()
    assert result == 3
    calls = [str(c) for c in mock_cur.execute.call_args_list]
    assert any("INSERT INTO attendance_archive" in c for c in calls)
    assert any("DELETE FROM attendance_log" in c for c in calls)


def test_archive_old_attendance_nothing_to_archive(mock_db_conn, mocker):
    """archive_old_attendance skips DELETE when nothing to archive."""
    mocker.patch("logger.config.SESSION_DURATION_MINUTES", 90)
    _mock_connect, mock_cur = mock_db_conn
    mock_cur.rowcount = 0
    result = archive_old_attendance()
    assert result == 0
    calls = [str(c) for c in mock_cur.execute.call_args_list]
    assert not any("DELETE FROM attendance_log" in c for c in calls)
