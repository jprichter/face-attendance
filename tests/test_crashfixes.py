from unittest.mock import MagicMock

import numpy as np

import logger
import monitor


def test_materialize_image_returns_contiguous_uint8_copy():
    source = np.linspace(0, 255, 5 * 5 * 3, dtype=np.float32).reshape((5, 5, 3))
    non_contiguous = source[:, ::2, :]

    materialized = monitor._materialize_image(non_contiguous)

    assert materialized.dtype == np.uint8
    assert materialized.flags["C_CONTIGUOUS"]
    assert materialized.shape == non_contiguous.shape

    original_value = materialized[0, 0, 0]
    non_contiguous[0, 0, 0] = 0
    assert materialized[0, 0, 0] == original_value


def test_handle_recognition_materializes_member_photo_before_write(mocker):
    mocker.patch("monitor.query_database", return_value=(101, "Test User", 0.1))
    mocker.patch("monitor.log_check_in", return_value=True)
    mocker.patch("monitor.save_member_image")
    mocker.patch("monitor.os.path.exists", return_value=False)
    mock_imwrite = mocker.patch("monitor.cv2.imwrite", return_value=True)
    mocker.patch("monitor.os.path.join", return_value="data/members/member_101.jpg")
    mocker.patch("monitor.config.CONFIRMATION_FRAMES", 1)

    confirmation_buffer = {"id": None, "count": 0}
    image = np.linspace(0, 255, 10 * 10 * 3, dtype=np.float32).reshape((10, 10, 3))[:, ::2, :]

    monitor.handle_recognition([0.1] * 512, confirmation_buffer, image)

    saved_image = mock_imwrite.call_args[0][1]
    assert saved_image.dtype == np.uint8
    assert saved_image.flags["C_CONTIGUOUS"]


def test_ensure_schema_adds_compatibility_columns():
    conn = MagicMock()
    cur = conn.cursor.return_value
    logger._SCHEMA_READY = False

    logger.ensure_schema(conn)

    executed_sql = [call[0][0] for call in cur.execute.call_args_list]
    assert any("ALTER TABLE members" in sql and "image_path" in sql for sql in executed_sql)
    assert any("ALTER TABLE unknown_detections" in sql and "image_path" in sql for sql in executed_sql)
    assert any("ALTER TABLE unknown_detections" in sql and "group_id" in sql for sql in executed_sql)
    conn.commit.assert_called_once()
    cur.close.assert_called_once()
