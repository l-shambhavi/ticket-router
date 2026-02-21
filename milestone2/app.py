"""
app.py
======
FastAPI entry point for Milestone 2.

WHY /status ENDPOINT:
  Without it you can only submit tickets — you can never show the result.
  actually ran. GET /status/{ticket_id} reads from Redis after Celery processes.

Endpoints:
  GET  /                       → service info
  GET  /health                 → redis + celery status
  POST /submit                 → 202 Accepted, queues to Celery
  GET  /status/{ticket_id}     → result after processing
"""
import json
import redis as sync_redis
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from milestone2.config import REDIS_URL
from milestone2.worker import celery_app, process_ticket

app = FastAPI(title="SmartSupport — Milestone 2", version="2.0.0")

_redis = sync_redis.from_url(REDIS_URL)


# ── Schema ────────────────────────────────────────────────────────────────────
class Ticket(BaseModel):
    ticket_id: str
    text: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "SmartSupport Milestone 2",
        "docs":    "/docs",
        "submit":  "POST /submit",
        "status":  "GET  /status/{ticket_id}",
        "health":  "GET  /health",
    }


@app.post("/submit", status_code=status.HTTP_202_ACCEPTED)
def submit(ticket: Ticket):
    """
    Returns 202 immediately.
    Celery picks up the task from Redis and processes it asynchronously.
    Atomic lock inside worker prevents duplicate processing.
    """
    task = process_ticket.delay(ticket.ticket_id, ticket.text)
    return {
        "status":    "accepted",
        "ticket_id": ticket.ticket_id,
        "task_id":   task.id,
    }


@app.get("/status/{ticket_id}")
def get_status(ticket_id: str):
    """
    Returns the classification result once Celery has processed the ticket.
    Raises 404 if not yet processed or ticket_id is invalid.
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
    redis_ok  = False
    celery_ok = False

    try:
        _redis.ping()
        redis_ok = True
    except Exception:
        pass

    try:
        inspector = celery_app.control.inspect(timeout=1)
        celery_ok = bool(inspector.ping())
    except Exception:
        pass

    return {"api": "ok", "redis": redis_ok, "celery": celery_ok}
