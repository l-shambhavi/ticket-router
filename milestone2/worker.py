"""
worker.py  (Milestone 4 — Circuit Breaker + Skill-Based Routing)
=================================================================
"""
import os
os.environ["HF_HOME"] = "./hf_cache"

import json
import redis
import threading
from celery import Celery
from datetime import datetime, timedelta
import numpy as np

from milestone2.config import REDIS_URL, URGENCY_THRESHOLD
from milestone2.classifier import classify as transformer_classify, urgency_score
from milestone2.webhook import send_alert
from milestone4.circuit_breaker import get_breaker
from milestone4.router import get_registry

from celery.utils.log import get_task_logger
logger = get_task_logger(__name__)

# ── Celery ────────────────────────────────────────────────────────────────────
celery_app = Celery("m4_worker", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)

# ── Redis ─────────────────────────────────────────────────────────────────────
redis_client   = redis.from_url(REDIS_URL)
embedding_lock = threading.Lock()
RECENT_KEY     = "recent_ticket_embeddings"


# ── Lock helpers ──────────────────────────────────────────────────────────────
def acquire_lock(ticket_id: str, ttl: int = 60) -> bool:
    return redis_client.set(f"lock:{ticket_id}", "1", nx=True, ex=ttl) is True

def release_lock(ticket_id: str):
    redis_client.delete(f"lock:{ticket_id}")


# ── Cosine / storm detection ──────────────────────────────────────────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_recent_tickets():
    raw = redis_client.get(RECENT_KEY)
    return json.loads(raw) if raw else []

def save_recent_tickets(tickets):
    redis_client.setex(RECENT_KEY, 3600, json.dumps(tickets))

def check_ticket_storm(embedding) -> int:
    now    = datetime.utcnow()
    cutoff = (now - timedelta(minutes=5)).isoformat()

    with embedding_lock:
        recent = get_recent_tickets()
        recent = [t for t in recent if t["timestamp"] > cutoff]
        similar_count = sum(
            1 for t in recent
            if cosine_similarity(np.array(t["embedding"]), embedding) > 0.9
        )
        recent.append({"embedding": embedding.tolist(), "timestamp": now.isoformat()})
        save_recent_tickets(recent)
        logger.info(f"[Ticket Storm] {similar_count} similar tickets in last 5 min")
    return similar_count


# ── Celery task ───────────────────────────────────────────────────────────────
@celery_app.task(name="process_ticket")
def process_ticket(ticket_id: str, text: str) -> dict:
    if not acquire_lock(ticket_id):
        logger.warning(f"Duplicate ticket skipped: {ticket_id}")
        return {"status": "duplicate", "ticket_id": ticket_id}

    try:
        # ── Circuit-breaker classification ────────────────────────────────
        breaker = get_breaker()
        category, model_used = breaker.classify(text, transformer_classify)
        logger.info(
            f"[CB] ticket={ticket_id} category={category} "
            f"model={model_used} state={breaker.state.value}"
        )

        # ── Urgency ───────────────────────────────────────────────────────
        urgency = urgency_score(text)

        # ── Embedding + storm ─────────────────────────────────────────────
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        embedding       = embedding_model.encode(text)
        similar_count   = check_ticket_storm(embedding)

        # ── Master incident ───────────────────────────────────────────────
        if similar_count > 10:
            master_incident = {
                "status":                     "master_incident_created",
                "ticket_id":                  ticket_id,
                "category":                   category,
                "model_used":                 model_used,
                "similar_tickets_last_5_min": similar_count,
                "circuit_breaker_state":      breaker.state.value,
            }
            redis_client.setex(f"result:{ticket_id}", 3600, json.dumps(master_incident))
            logger.info(f"Master Incident CREATED for ticket {ticket_id}")
            return master_incident

        # ── Alert ─────────────────────────────────────────────────────────
        if urgency > URGENCY_THRESHOLD:
            send_alert(ticket_id, urgency, category)
            logger.info(f"Alert sent: {ticket_id} urgency={urgency}")

        # ── Skill-based routing ───────────────────────────────────────────
        registry = get_registry()
        decision = registry.route(ticket_id, category)
        logger.info(
            f"[Router] ticket={ticket_id} → agent={decision.agent_name} "
            f"score={decision.score:.4f} reason='{decision.reason}'"
        )

        result = {
            "status":                "done",
            "ticket_id":             ticket_id,
            "category":              category,
            "urgency_score":         urgency,
            "model_used":            model_used,
            "circuit_breaker_state": breaker.state.value,
            "routing": {
                "agent_id":   decision.agent_id,
                "agent_name": decision.agent_name,
                "score":      round(decision.score, 4),
                "reason":     decision.reason,
            } if decision.agent_id else {
                "agent_id": None,
                "reason":   decision.reason,
            },
        }

        # ── Save monitoring data to Redis ─────────────────────────────────
        redis_client.setex("breaker_stats", 3600, json.dumps(breaker.stats()))

        history_raw = redis_client.get("routing_history")
        history = json.loads(history_raw) if history_raw else []
        history.append(decision.to_dict())
        redis_client.setex("routing_history", 3600, json.dumps(history[-100:]))

        # ── Save agent registry snapshot ──────────────────────────────────
        redis_client.setex(
            "agent_registry_snapshot", 3600,
            json.dumps(registry.all_agents())
        )

        # ── Save result ───────────────────────────────────────────────────
        redis_client.setex(f"result:{ticket_id}", 3600, json.dumps(result))
        logger.info(f"Ticket {ticket_id} done — {similar_count} similar in last 5 min")
        return result

    finally:
        release_lock(ticket_id)