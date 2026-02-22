"""
app.py
======
FastAPI entry point for Milestone 2 + Milestone 3 integration.

Endpoints:
  GET  /                       → service info
  GET  /health                 → Redis + Celery status
  POST /submit                 → 202 Accepted, queues to Celery
  GET  /status/{ticket_id}     → result after processing
"""

import json
import redis as sync_redis
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from milestone2.config import REDIS_URL
from milestone2.worker import celery_app, process_ticket

app = FastAPI(title="SmartSupport — Milestone 2/3", version="3.0.0")

# Redis client to store results
_redis = sync_redis.from_url(REDIS_URL)


# ── Ticket Schema ─────────────────────────────────────────────────────────────
class Ticket(BaseModel):
    ticket_id: str
    text: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "SmartSupport Milestone 2/3",
        "docs": "/docs",
        "submit": "POST /submit",
        "status": "GET /status/{ticket_id}",
        "health": "GET /health",
    }


@app.post("/submit", status_code=status.HTTP_202_ACCEPTED)
def submit(ticket: Ticket):
    """
    Submit a ticket for async processing.
    
    Returns 202 immediately.
    Celery worker will handle:
      - Category classification
      - Urgency scoring
      - Semantic deduplication (Milestone 3)
      - Ticket storm detection / Master Incident creation
    """
    task = process_ticket.delay(ticket.ticket_id, ticket.text)
    return {
        "status": "accepted",
        "ticket_id": ticket.ticket_id,
        "task_id": task.id,
    }


@app.get("/status/{ticket_id}")
def get_status(ticket_id: str):
    """
    Returns the processed ticket result from Redis.
    Raises 404 if ticket is not yet processed or invalid ID.
    """
    raw = _redis.get(f"result:{ticket_id}")
    if not raw:
        raise HTTPException(
            status_code=404,
            detail="Result not found — ticket still queued or invalid ID."
        )
    return json.loads(raw.decode())


@app.get("/health")
def health():
    """
    Returns the health status of API, Redis, and Celery.
    """
    redis_ok = False
    celery_ok = False

    # Check Redis
    try:
        _redis.ping()
        redis_ok = True
    except Exception:
        pass

    # Check Celery
    try:
        inspector = celery_app.control.inspect(timeout=1)
        celery_ok = bool(inspector.ping())
    except Exception:
        pass

    return {"api": "ok", "redis": redis_ok, "celery": celery_ok}