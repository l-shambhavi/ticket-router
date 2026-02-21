"""
test_worker.py
==============
Tests the Celery task logic 
We call process_ticket.run() directly (bypasses Celery queue entirely).
"""
import pytest
from unittest.mock import patch, MagicMock


# ── Helper: common patches applied to every test ──────────────────────────────
BASE_PATCHES = [
    "milestone2.worker.acquire_lock",
    "milestone2.worker.release_lock",
    "milestone2.worker.classify",
    "milestone2.worker.urgency_score",
    "milestone2.worker.send_alert",
    "milestone2.worker.redis_client",
]


def test_high_urgency_ticket_processed_and_alert_sent():
    with patch("milestone2.worker.acquire_lock",  return_value=True), \
         patch("milestone2.worker.release_lock"), \
         patch("milestone2.worker.classify",      return_value="Technical"), \
         patch("milestone2.worker.urgency_score", return_value=0.92), \
         patch("milestone2.worker.send_alert")    as mock_alert, \
         patch("milestone2.worker.redis_client"):

        from milestone2.worker import process_ticket
        result = process_ticket.run("tid-001", "System completely down ASAP!")

    assert result["status"]        == "done"
    assert result["category"]      == "Technical"
    assert result["urgency_score"] == 0.92
    mock_alert.assert_called_once_with("tid-001", 0.92, "Technical")


def test_low_urgency_ticket_no_alert():
    with patch("milestone2.worker.acquire_lock",  return_value=True), \
         patch("milestone2.worker.release_lock"), \
         patch("milestone2.worker.classify",      return_value="Billing"), \
         patch("milestone2.worker.urgency_score", return_value=0.3), \
         patch("milestone2.worker.send_alert")    as mock_alert, \
         patch("milestone2.worker.redis_client"):

        from milestone2.worker import process_ticket
        result = process_ticket.run("tid-002", "Can I get a billing statement?")

    assert result["status"] == "done"
    mock_alert.assert_not_called()


def test_duplicate_ticket_rejected():
    with patch("milestone2.worker.acquire_lock",  return_value=False), \
         patch("milestone2.worker.redis_client"):

        from milestone2.worker import process_ticket
        result = process_ticket.run("tid-dup", "any text")

    assert result["status"]    == "duplicate"
    assert result["ticket_id"] == "tid-dup"


def test_result_stored_in_redis():
    mock_redis = MagicMock()

    with patch("milestone2.worker.acquire_lock",  return_value=True), \
         patch("milestone2.worker.release_lock"), \
         patch("milestone2.worker.classify",      return_value="Legal"), \
         patch("milestone2.worker.urgency_score", return_value=0.5), \
         patch("milestone2.worker.send_alert"), \
         patch("milestone2.worker.redis_client",  mock_redis):

        from milestone2.worker import process_ticket
        process_ticket.run("tid-003", "Legal query")

    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args[0]
    assert args[0] == "result:tid-003"
    assert args[1] == 3600


def test_lock_always_released_even_on_error():
    with patch("milestone2.worker.acquire_lock",  return_value=True), \
         patch("milestone2.worker.release_lock")  as mock_release, \
         patch("milestone2.worker.classify",      side_effect=RuntimeError("model crashed")), \
         patch("milestone2.worker.redis_client"):

        from milestone2.worker import process_ticket
        try:
            process_ticket.run("tid-err", "some text")
        except RuntimeError:
            pass

    mock_release.assert_called_once_with("tid-err")
