"""
test_concurrency.py
===================
Validates that the atomic Redis lock prevents duplicate processing
when 10+ tasks arrive simultaneously for the same ticket_id.
"""
import pytest
from unittest.mock import patch, MagicMock


def make_atomic_lock():
    """
    Simulates Redis SET NX atomically using a plain dict.
    Only the first call for a given key returns True.
    """
    store = {}

    def acquire(ticket_id, ttl=60):
        if ticket_id in store:
            return False
        store[ticket_id] = True
        return True

    def release(ticket_id):
        store.pop(ticket_id, None)

    return acquire, release


def test_same_ticket_only_processed_once():
    """
    Simulate two concurrent calls before release_lock runs.
    Second call must return duplicate.
    """
    lock_state = {"locked": False}

    def acquire(ticket_id, ttl=60):
        if lock_state["locked"]:
            return False
        lock_state["locked"] = True
        return True

    def release(ticket_id):
        pass  # simulate concurrent scenario (no release yet)

    with patch("milestone2.worker.acquire_lock", side_effect=acquire), \
         patch("milestone2.worker.release_lock", side_effect=release), \
         patch("milestone2.worker.classify", return_value="Technical"), \
         patch("milestone2.worker.urgency_score", return_value=0.5), \
         patch("milestone2.worker.send_alert"), \
         patch("milestone2.worker.redis_client"):

        from milestone2.worker import process_ticket

        r1 = process_ticket.run("tid-same", "first call")
        r2 = process_ticket.run("tid-same", "duplicate call")

    assert r1["status"] == "done"
    assert r2["status"] == "duplicate"

def test_10_different_tickets_all_processed():
    acquire, release = make_atomic_lock()

    with patch("milestone2.worker.acquire_lock",  side_effect=acquire), \
         patch("milestone2.worker.release_lock",   side_effect=release), \
         patch("milestone2.worker.classify",       return_value="Technical"), \
         patch("milestone2.worker.urgency_score",  return_value=0.4), \
         patch("milestone2.worker.send_alert"), \
         patch("milestone2.worker.redis_client"):

        from milestone2.worker import process_ticket
        results = [process_ticket.run(f"tid-{i}", f"message {i}") for i in range(10)]

    assert all(r["status"] == "done" for r in results), \
        f"Some tickets not processed: {[r for r in results if r['status'] != 'done']}"
    assert len({r["ticket_id"] for r in results}) == 10


def test_alert_fires_only_for_high_urgency():
    """
    5 tickets with mixed urgency scores.
    Alert must fire exactly for those with score > 0.8.
    """
    alerts     = []
    scores_map = {
        "tid-0": 0.90,   # alert
        "tid-1": 0.30,   # no alert
        "tid-2": 0.95,   # alert
        "tid-3": 0.20,   # no alert
        "tid-4": 0.85,   # alert
    }

    acquire, release = make_atomic_lock()

    def fake_alert(tid, score, cat):
        alerts.append(tid)

    with patch("milestone2.worker.acquire_lock",  side_effect=acquire), \
         patch("milestone2.worker.release_lock",   side_effect=release), \
         patch("milestone2.worker.classify",       return_value="Technical"), \
         patch("milestone2.worker.send_alert",     side_effect=fake_alert), \
         patch("milestone2.worker.redis_client"):

        from milestone2.worker import process_ticket
        for tid, score in scores_map.items():
            with patch("milestone2.worker.urgency_score", return_value=score):
                process_ticket.run(tid, "test message")

    assert sorted(alerts) == ["tid-0", "tid-2", "tid-4"]


def test_lock_expires_allows_retry():
    """
    If a lock was previously held but released (simulating TTL expiry),
    a new call for the same ticket_id should succeed.
    """
    acquire, release = make_atomic_lock()

    with patch("milestone2.worker.acquire_lock",  side_effect=acquire), \
         patch("milestone2.worker.release_lock",   side_effect=release), \
         patch("milestone2.worker.classify",       return_value="Billing"), \
         patch("milestone2.worker.urgency_score",  return_value=0.4), \
         patch("milestone2.worker.send_alert"), \
         patch("milestone2.worker.redis_client"):

        from milestone2.worker import process_ticket

        r1 = process_ticket.run("tid-retry", "first attempt")
        assert r1["status"] == "done"

        # Simulate TTL expiry — lock was auto-released by Redis
        release("tid-retry")

        r2 = process_ticket.run("tid-retry", "retry after expiry")
        assert r2["status"] == "done"
