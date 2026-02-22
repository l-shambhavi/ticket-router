"""
app.py  (Milestone 4 — Circuit Breaker + Skill-Based Routing)
=============================================================
"""

import json
import redis as sync_redis
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Optional

from milestone2.config import REDIS_URL
from milestone2.worker import celery_app, process_ticket

app = FastAPI(title="SmartSupport — Milestone 4", version="4.0.0")
_redis = sync_redis.from_url(REDIS_URL)


# ── Schemas ───────────────────────────────────────────────────────────────────
class Ticket(BaseModel):
    ticket_id: str
    text: str

class AgentCreate(BaseModel):
    agent_id:     str
    name:         str
    skill_vector: Dict[str, float]
    max_capacity: int = 5


# ── Core ticket endpoints ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "SmartSupport Milestone 4",
        "docs":    "/docs",
        "endpoints": [
            "POST /submit",
            "GET  /status/{ticket_id}",
            "GET  /health",
            "GET  /breaker/stats",
            "GET  /agents",
            "GET  /routing/stats",
            "GET  /routing/recent",
        ],
    }

@app.post("/submit", status_code=status.HTTP_202_ACCEPTED)
def submit(ticket: Ticket):
    task = process_ticket.delay(ticket.ticket_id, ticket.text)
    return {"status": "accepted", "ticket_id": ticket.ticket_id, "task_id": task.id}

@app.get("/status/{ticket_id}")
def get_status(ticket_id: str):
    raw = _redis.get(f"result:{ticket_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Not found or still queued.")
    return json.loads(raw.decode())

@app.get("/health")
def health():
    redis_ok, celery_ok = False, False
    try:
        _redis.ping()
        redis_ok = True
    except Exception:
        pass
    try:
        celery_ok = bool(celery_app.control.inspect(timeout=1).ping())
    except Exception:
        pass
    return {"api": "ok", "redis": redis_ok, "celery": celery_ok}


# ── Circuit breaker ───────────────────────────────────────────────────────────
@app.get("/breaker/stats")
def breaker_stats():
    """Read circuit breaker stats saved by the worker into Redis."""
    raw = _redis.get("breaker_stats")
    if raw:
        return json.loads(raw)
    return {
        "state": "CLOSED",
        "note": "No stats yet — submit a ticket first"
    }


# ── Agent registry ────────────────────────────────────────────────────────────
@app.get("/agents")
def list_agents():
    """Read agent snapshot saved by the worker into Redis."""
    raw = _redis.get("agent_registry_snapshot")
    if raw:
        return {"agents": json.loads(raw)}
    return {"agents": [], "note": "No snapshot yet — submit a ticket first"}


# ── Routing analytics ─────────────────────────────────────────────────────────
@app.get("/routing/recent")
def routing_recent(n: int = 20):
    """Read routing history saved by the worker into Redis."""
    raw = _redis.get("routing_history")
    if raw:
        decisions = json.loads(raw)
        return {"decisions": decisions[-n:]}
    return {"decisions": [], "note": "No decisions yet — submit a ticket first"}

@app.get("/routing/stats")
def routing_stats():
    """Summarise routing history from Redis."""
    raw = _redis.get("routing_history")
    if not raw:
        return {"total_routed": 0, "note": "No decisions yet"}

    decisions = json.loads(raw)
    total     = len(decisions)
    unrouted  = sum(1 for d in decisions if not d.get("agent_id"))
    by_cat    = {}
    by_agent  = {}

    for d in decisions:
        cat = d.get("category", "Unknown")
        by_cat[cat] = by_cat.get(cat, 0) + 1
        name = d.get("agent_name")
        if name:
            by_agent[name] = by_agent.get(name, 0) + 1

    avg_score = (
        sum(d.get("score", 0) for d in decisions if d.get("agent_id"))
        / max(1, total - unrouted)
    )

    return {
        "total_routed":  total,
        "unrouted":      unrouted,
        "by_category":   by_cat,
        "by_agent":      by_agent,
        "avg_score":     round(avg_score, 4),
    }