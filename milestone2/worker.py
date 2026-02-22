"""
worker.py
=========

Celery worker — processes tickets asynchronously from Redis queue.

Changes for Milestone 3:
- Celery logger used for proper logging
- Ticket storm counts logged for debugging
- Master incident creation clearly logged
"""
import os
os.environ["HF_HOME"] = "./hf_cache"

import redis
import json
from celery import Celery
from datetime import datetime, timedelta
import numpy as np
import threading

from milestone2.config import REDIS_URL, URGENCY_THRESHOLD
from milestone2.classifier import classify, urgency_score
from milestone2.webhook import send_alert

# ── Celery logger setup ───────────────────────────────────────────────────────
from celery.utils.log import get_task_logger
logger = get_task_logger(__name__)

# ── Celery app ───────────────────────────────────────────────────────────────
celery_app = Celery("m2_worker", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)

# Redis client for locks, result storage, and recent embeddings
redis_client = redis.from_url(REDIS_URL)

# Lock for thread safety
embedding_lock = threading.Lock()

# Redis key to store recent embeddings
RECENT_KEY = "recent_ticket_embeddings"


# ── Atomic lock helpers ───────────────────────────────────────────────────────
def acquire_lock(ticket_id: str, ttl: int = 60) -> bool:
    """Atomic Redis lock for ticket_id"""
    result = redis_client.set(f"lock:{ticket_id}", "1", nx=True, ex=ttl)
    return result is True


def release_lock(ticket_id: str):
    redis_client.delete(f"lock:{ticket_id}")


# ── Cosine similarity / ticket storm ──────────────────────────────────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_recent_tickets():
    """Load recent tickets from Redis"""
    raw = redis_client.get(RECENT_KEY)
    if raw:
        return json.loads(raw)
    return []


def save_recent_tickets(tickets):
    redis_client.setex(RECENT_KEY, 3600, json.dumps(tickets))


def check_ticket_storm(embedding) -> int:
    """
    Returns number of highly similar tickets (>0.9) in last 5 min
    Stores embedding in Redis for persistence
    """
    now = datetime.utcnow()
    cutoff = (now - timedelta(minutes=5)).isoformat()

    with embedding_lock:
        recent_ticket_embeddings = get_recent_tickets()

        # Remove old tickets
        recent_ticket_embeddings = [
            t for t in recent_ticket_embeddings if t["timestamp"] > cutoff
        ]

        # Count similar tickets
        similar_count = sum(
            1 for t in recent_ticket_embeddings if cosine_similarity(np.array(t["embedding"]), embedding) > 0.9
        )

        # Add current ticket
        recent_ticket_embeddings.append({"embedding": embedding.tolist(), "timestamp": now.isoformat()})
        save_recent_tickets(recent_ticket_embeddings)

        # Debug log to track ticket storm count
        logger.info(f"[Ticket Storm] {similar_count} similar tickets in last 5 min")

    return similar_count


# ── Celery task ───────────────────────────────────────────────────────────────
@celery_app.task(name="process_ticket")
def process_ticket(ticket_id: str, text: str) -> dict:
    """
    1. Acquire atomic lock
    2. Classify category
    3. Score urgency
    4. Detect ticket storm
    5. If storm → create master incident
    6. Else → fire alert if urgent
    7. Store result in Redis
    8. Release lock
    """
    if not acquire_lock(ticket_id):
        logger.warning(f"Duplicate ticket skipped: {ticket_id}")
        return {"status": "duplicate", "ticket_id": ticket_id}

    try:
        # ── Load model inside task (Windows-friendly) ────────────────
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        # ── Classification & urgency ───────────────────────────────
        category = classify(text)
        urgency = urgency_score(text)

        # ── Compute embedding & storm detection ────────────────────
        embedding = embedding_model.encode(text)
        similar_count = check_ticket_storm(embedding)

        # ── Master incident ────────────────────────────────────────
        if similar_count > 10:
            master_incident = {
                "status": "master_incident_created",
                "ticket_id": ticket_id,
                "category": category,
                "similar_tickets_last_5_min": similar_count,
            }
            redis_client.setex(f"result:{ticket_id}", 3600, json.dumps(master_incident))
            logger.info(f"Master Incident CREATED for ticket {ticket_id} — {similar_count} similar tickets")
            return master_incident

        # ── Normal flow ────────────────────────────────────────────
        if urgency > URGENCY_THRESHOLD:
            send_alert(ticket_id, urgency, category)
            logger.info(f"Alert sent for urgent ticket {ticket_id} — urgency {urgency}")

        result = {
            "status": "done",
            "ticket_id": ticket_id,
            "category": category,
            "urgency_score": urgency,
        }
        redis_client.setex(f"result:{ticket_id}", 3600, json.dumps(result))
        logger.info(f"Ticket {ticket_id} processed successfully — {similar_count} similar tickets in last 5 min")

        return result

    finally:
        release_lock(ticket_id)