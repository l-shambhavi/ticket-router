"""
app.py  (Milestone 4 — Circuit Breaker + Skill-Based Routing)
=============================================================

New endpoints:
  GET  /breaker/stats          → circuit breaker metrics
  GET  /agents                 → all agent skill vectors + current load
  POST /agents                 → register a new agent
  DELETE /agents/{agent_id}    → deregister agent
  POST /agents/{agent_id}/release → decrement agent load (ticket resolved)
  GET  /routing/stats          → routing history summary
  GET  /routing/recent         → last N routing decisions
"""

import json
import redis as sync_redis
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Optional

from milestone2.config import REDIS_URL
from milestone2.worker import celery_app, process_ticket
from milestone4.circuit_breaker import get_breaker
from milestone4.router import get_registry, Agent

app = FastAPI(title="SmartSupport — Milestone 4", version="4.0.0")
_redis = sync_redis.from_url(REDIS_URL)


# ── Schemas ───────────────────────────────────────────────────────────────────
class Ticket(BaseModel):
    ticket_id: str
    text: str

class AgentCreate(BaseModel):
    agent_id:     str
    name:         str
    skill_vector: Dict[str, float]   # e.g. {"Technical": 0.9, "Billing": 0.1}
    max_capacity: int = 5


# ── Core ticket endpoints ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "SmartSupport Milestone 4",
        "docs":    "/docs",
        "new":     ["/breaker/stats", "/agents", "/routing/stats", "/routing/recent"],
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
        _redis.ping(); redis_ok = True
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
    """Current circuit breaker state and latency metrics."""
    return get_breaker().stats()

@app.post("/breaker/reset")
def breaker_reset():
    """Manually force breaker back to CLOSED (for ops use)."""
    b = get_breaker()
    import milestone2.circuit_breaker as cb_mod
    b._state         = cb_mod.BreakerState.CLOSED if hasattr(cb_mod, "BreakerState") else b._state
    b._failure_count = 0
    return {"message": "Circuit breaker reset to CLOSED"}


# ── Agent registry ────────────────────────────────────────────────────────────
@app.get("/agents")
def list_agents():
    """List all registered agents with their skill vectors and current load."""
    return {"agents": get_registry().all_agents()}

@app.post("/agents", status_code=status.HTTP_201_CREATED)
def register_agent(payload: AgentCreate):
    """Register a new agent."""
    agent = Agent(
        agent_id     = payload.agent_id,
        name         = payload.name,
        skill_vector = payload.skill_vector,
        max_capacity = payload.max_capacity,
    )
    get_registry().register(agent)
    return {"message": f"Agent '{payload.name}' registered.", "agent": agent.to_dict()}

@app.delete("/agents/{agent_id}")
def deregister_agent(agent_id: str):
    ok = get_registry().deregister(agent_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return {"message": f"Agent {agent_id} removed."}

@app.post("/agents/{agent_id}/release")
def release_agent_slot(agent_id: str):
    """Mark one ticket as resolved, freeing a slot on the agent."""
    ok = get_registry().release(agent_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Agent not found or load already 0.")
    agent = get_registry().get_agent(agent_id)
    return {"message": "Slot released.", "current_load": agent.current_load if agent else None}

@app.patch("/agents/{agent_id}/active")
def set_agent_active(agent_id: str, active: bool = True):
    get_registry().set_active(agent_id, active)
    return {"message": f"Agent {agent_id} active={active}"}


# ── Routing analytics ─────────────────────────────────────────────────────────
@app.get("/routing/stats")
def routing_stats():
    return get_registry().routing_stats()

@app.get("/routing/recent")
def routing_recent(n: int = 20):
    return {"decisions": get_registry().recent_decisions(n)}