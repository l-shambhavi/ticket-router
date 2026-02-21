"""
worker.py
=========
Celery worker — processes tickets asynchronously from Redis queue.

HOW ATOMIC LOCKING WORKS:
  Redis SET with NX (set if not exists) + EX (TTL) is a single atomic command.
  Even if 10+ tasks for the same ticket_id arrive at the same millisecond,
  only the first SET succeeds. The rest return None → "duplicate" response.
  The TTL (60s) means the lock auto-expires if the worker crashes mid-task,
  so no ticket stays permanently stuck.

"""
import redis
import json
from celery import Celery
from milestone2.config     import REDIS_URL, URGENCY_THRESHOLD
from milestone2.classifier import classify, urgency_score
from milestone2.webhook    import send_alert

# ── Celery app ────────────────────────────────────────────────────────────────
celery_app = Celery("m2_worker", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer           = "json",
    result_serializer         = "json",
    accept_content            = ["json"],
    task_track_started        = True,
    worker_prefetch_multiplier = 1,   # process one task at a time per worker
)

# Sync Redis client for locks + result storage
redis_client = redis.from_url(REDIS_URL)


# ── Atomic lock helpers ───────────────────────────────────────────────────────
def acquire_lock(ticket_id: str, ttl: int = 60) -> bool:
    """
    SET lock:{ticket_id} 1 NX EX 60
    Atomic — returns True only for the first caller, False for all others.
    TTL ensures the lock is released even if the worker crashes.
    """
    result = redis_client.set(f"lock:{ticket_id}", "1", nx=True, ex=ttl)
    return result is True


def release_lock(ticket_id: str):
    redis_client.delete(f"lock:{ticket_id}")


# ── Celery task ───────────────────────────────────────────────────────────────
@celery_app.task(name="process_ticket")
def process_ticket(ticket_id: str, text: str) -> dict:
    """
    1. Try to acquire atomic lock  → reject duplicates immediately
    2. Classify category           → Billing / Technical / Legal
    3. Score urgency               → S ∈ [0, 1]
    4. Fire Discord alert          → only if S > URGENCY_THRESHOLD (0.8)
    5. Store result in Redis       → readable by GET /status/{ticket_id}
    6. Release lock (always)
    """
    if not acquire_lock(ticket_id):
        return {"status": "duplicate", "ticket_id": ticket_id}

    try:
        category = classify(text)
        urgency  = urgency_score(text)

        if urgency > URGENCY_THRESHOLD:
            send_alert(ticket_id, urgency, category)

        result = {
            "status":        "done",
            "ticket_id":     ticket_id,
            "category":      category,
            "urgency_score": urgency,
        }

        # Store for 1 hour so status can retrieve it
        redis_client.setex(f"result:{ticket_id}", 3600, json.dumps(result))
        return result

    finally:
        release_lock(ticket_id)
