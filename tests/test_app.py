import pytest
from datetime import datetime

from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index_route(client, mocker):
    mocker.patch("app.get_attendance_logs", return_value=[
        ("data/members/member_1.jpg", "John Doe", datetime(2023, 10, 27, 10, 0, 0)),
        ("data/members/member_2.jpg", "Jane Smith", datetime(2023, 10, 27, 10, 5, 0)),
    ])
    mocker.patch("app.get_unknown_groups", return_value=[])

    response = client.get('/')
    assert response.status_code == 200
    assert b"John Doe" in response.data
    assert b"Jane Smith" in response.data
    assert b"Face Attendance" in response.data


def test_index_no_logs(client, mocker):
    mocker.patch("app.get_attendance_logs", return_value=[])
    mocker.patch("app.get_unknown_groups", return_value=[])

    response = client.get('/')
    assert response.status_code == 200
    assert b"No logs found" in response.data


def test_enroll_success(client, mocker):
    mocker.patch("app.enroll_from_unknown", return_value=42)

    response = client.post('/enroll',
        json={"group_id": "abc-123", "name": "John Doe"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["member_id"] == 42
    assert data["name"] == "John Doe"


def test_enroll_missing_name(client, mocker):
    response = client.post('/enroll', json={"group_id": "abc-123"})
    assert response.status_code == 400


def test_enroll_missing_group_id(client, mocker):
    response = client.post('/enroll', json={"name": "John Doe"})
    assert response.status_code == 400


def test_enroll_group_not_found(client, mocker):
    mocker.patch("app.enroll_from_unknown", return_value=None)

    response = client.post('/enroll',
        json={"group_id": "nonexistent", "name": "John Doe"})

    assert response.status_code == 404


def test_dismiss_success(client, mocker):
    mocker.patch("app.dismiss_unknown_group", return_value=True)

    response = client.post('/dismiss', json={"group_id": "abc-123"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True


def test_dismiss_missing_group_id(client, mocker):
    response = client.post('/dismiss', json={})
    assert response.status_code == 400


def test_dismiss_failure(client, mocker):
    mocker.patch("app.dismiss_unknown_group", return_value=False)

    response = client.post('/dismiss', json={"group_id": "abc-123"})

    assert response.status_code == 500


def test_serve_member_photo(client, mocker, tmp_path):
    photo = tmp_path / "member_1.jpg"
    photo.write_bytes(b"\xff\xd8fake-jpeg")
    mocker.patch("app.config.MEMBER_FACES_DIR", str(tmp_path))

    response = client.get('/member-photos/member_1.jpg')
    assert response.status_code == 200
    assert response.data == b"\xff\xd8fake-jpeg"


def test_serve_member_photo_not_found(client, mocker, tmp_path):
    mocker.patch("app.config.MEMBER_FACES_DIR", str(tmp_path))

    response = client.get('/member-photos/nonexistent.jpg')
    assert response.status_code == 404


def test_index_renders_member_photo(client, mocker):
    mocker.patch("app.get_attendance_logs", return_value=[
        ("data/members/member_1.jpg", "John Doe", datetime(2023, 10, 27, 10, 0, 0)),
        (None, "Jane Smith", datetime(2023, 10, 27, 10, 5, 0)),
    ])
    mocker.patch("app.get_unknown_groups", return_value=[])

    response = client.get('/')
    assert response.status_code == 200
    assert b'/member-photos/member_1.jpg' in response.data
    assert b'text-muted' in response.data  # placeholder for Jane (no photo)


def test_index_with_unknown_groups(client, mocker):
    mocker.patch("app.get_attendance_logs", return_value=[])
    mocker.patch("app.get_unknown_groups", return_value=[
        ("group-uuid-1", "data/unknown/unknown_20231027_100000.jpg", 3,
         datetime(2023, 10, 27, 10, 0, 0), datetime(2023, 10, 27, 10, 15, 0)),
    ])

    response = client.get('/')
    assert response.status_code == 200
    assert b"Enroll" in response.data
    assert b"3 times" in response.data
