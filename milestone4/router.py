"""
router.py
=========
Skill-Based Ticket Router with Constraint Optimization.

Each agent has a Skill Vector:
    { "Technical": 0.9, "Billing": 0.1, "Legal": 0.0 }

Routing solves a linear assignment / greedy optimization:

    score(agent, ticket) = skill_match(agent, category)
                         × availability_factor(agent)

Subject to:
    - agent.current_load < agent.max_capacity
    - agent must support the required category (skill > 0)

The router picks the agent with the highest score. In the event of a tie
the agent with the lower current load wins.

Thread-safe: all mutations go through a single lock.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Data classes ──────────────────────────────────────────────────────────────

VALID_CATEGORIES = {"Technical", "Billing", "Legal"}


@dataclass
class Agent:
    """Represents a support agent with skills and capacity."""
    agent_id:     str
    name:         str
    skill_vector: Dict[str, float]   # category → proficiency [0, 1]
    max_capacity: int = 5            # max simultaneous tickets
    current_load: int = 0
    active:       bool = True

    # Audit fields
    total_handled: int = 0
    last_assigned: float = 0.0

    def __post_init__(self):
        # Normalise skill vector so values sum to 1
        total = sum(self.skill_vector.values())
        if total > 0:
            self.skill_vector = {k: v / total for k, v in self.skill_vector.items()}
        # Ensure all categories present
        for cat in VALID_CATEGORIES:
            self.skill_vector.setdefault(cat, 0.0)

    @property
    def available_slots(self) -> int:
        return max(0, self.max_capacity - self.current_load)

    @property
    def load_ratio(self) -> float:
        return self.current_load / self.max_capacity if self.max_capacity else 1.0

    def availability_factor(self) -> float:
        """
        Penalises heavily loaded agents.
        0 → fully loaded, 1 → completely free.
        Uses a smooth concave curve so near-full agents are strongly penalised.
        """
        return 1.0 - (self.load_ratio ** 0.5)

    def skill_match(self, category: str) -> float:
        return self.skill_vector.get(category, 0.0)

    def routing_score(self, category: str) -> float:
        """
        Composite score used by the optimiser.

        score = skill_match × availability_factor
        Range: [0, 1]
        """
        return self.skill_match(category) * self.availability_factor()

    def to_dict(self) -> dict:
        return {
            "agent_id":      self.agent_id,
            "name":          self.name,
            "skill_vector":  {k: round(v, 3) for k, v in self.skill_vector.items()},
            "max_capacity":  self.max_capacity,
            "current_load":  self.current_load,
            "available_slots": self.available_slots,
            "load_ratio":    round(self.load_ratio, 3),
            "total_handled": self.total_handled,
            "active":        self.active,
        }


@dataclass
class RoutingDecision:
    ticket_id:   str
    category:    str
    agent_id:    Optional[str]
    agent_name:  Optional[str]
    score:       float
    reason:      str
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "ticket_id":  self.ticket_id,
            "category":   self.category,
            "agent_id":   self.agent_id,
            "agent_name": self.agent_name,
            "score":      round(self.score, 4),
            "reason":     self.reason,
            "routed_at":  self.timestamp,
        }


# ── Registry & Router ─────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Stateful registry of agents + constraint-optimised routing.

    Constraint Optimisation (single-ticket assignment):
        maximise  score(a, ticket)
        subject to
            a.current_load < a.max_capacity        (capacity constraint)
            a.skill_match(category) > 0            (skill constraint)
            a.active == True                       (availability constraint)

    When multiple agents tie within SCORE_EPSILON, prefer the one with
    the lowest current_load (load balancing tie-break).
    """

    SCORE_EPSILON = 0.01   # ties within this margin → load-balance

    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._lock   = threading.Lock()
        self._history: List[RoutingDecision] = []

    # ── Agent CRUD ────────────────────────────────────────────────────────────

    def register(self, agent: Agent) -> Agent:
        with self._lock:
            self._agents[agent.agent_id] = agent
        return agent

    def deregister(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                return True
        return False

    def set_active(self, agent_id: str, active: bool):
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].active = active

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self._agents.get(agent_id)

    def all_agents(self) -> List[dict]:
        with self._lock:
            return [a.to_dict() for a in self._agents.values()]

    # ── Core optimiser ────────────────────────────────────────────────────────

    def _candidates(self, category: str) -> List[Agent]:
        """Return agents that satisfy hard constraints."""
        return [
            a for a in self._agents.values()
            if a.active
            and a.available_slots > 0
            and a.skill_match(category) > 0.0
        ]

    def _optimise(self, category: str) -> Tuple[Optional[Agent], float]:
        """
        Greedy constraint optimisation:
          1. Filter to feasible candidates.
          2. Rank by routing_score (skill × availability).
          3. Tie-break by current_load (ascending).
        Returns (best_agent, score) or (None, 0.0) if no candidates.
        """
        candidates = self._candidates(category)
        if not candidates:
            return None, 0.0

        # Sort descending by score, then ascending by load
        ranked = sorted(
            candidates,
            key=lambda a: (-a.routing_score(category), a.current_load)
        )

        best       = ranked[0]
        best_score = best.routing_score(category)

        # Among agents within SCORE_EPSILON of best, pick lowest load
        near_top = [
            a for a in ranked
            if (best_score - a.routing_score(category)) <= self.SCORE_EPSILON
        ]
        if len(near_top) > 1:
            best = min(near_top, key=lambda a: a.current_load)

        return best, best.routing_score(category)

    # ── Public routing API ────────────────────────────────────────────────────

    def route(self, ticket_id: str, category: str) -> RoutingDecision:
        """
        Assign ticket to the optimal available agent.
        Increments agent.current_load atomically.
        """
        with self._lock:
            agent, score = self._optimise(category)

            if agent is None:
                decision = RoutingDecision(
                    ticket_id  = ticket_id,
                    category   = category,
                    agent_id   = None,
                    agent_name = None,
                    score      = 0.0,
                    reason     = f"No available agent with skills for '{category}'",
                )
            else:
                agent.current_load  += 1
                agent.total_handled += 1
                agent.last_assigned  = time.time()

                decision = RoutingDecision(
                    ticket_id  = ticket_id,
                    category   = category,
                    agent_id   = agent.agent_id,
                    agent_name = agent.name,
                    score      = score,
                    reason     = (
                        f"skill_match={agent.skill_match(category):.3f}, "
                        f"availability={agent.availability_factor():.3f}, "
                        f"load={agent.current_load}/{agent.max_capacity}"
                    ),
                )

            self._history.append(decision)
            return decision

    def release(self, agent_id: str) -> bool:
        """
        Call when a ticket is resolved to free one slot on the agent.
        """
        with self._lock:
            agent = self._agents.get(agent_id)
            if agent and agent.current_load > 0:
                agent.current_load -= 1
                return True
        return False

    # ── Analytics ─────────────────────────────────────────────────────────────

    def routing_stats(self) -> dict:
        with self._lock:
            total   = len(self._history)
            unrouted = sum(1 for d in self._history if d.agent_id is None)
            by_cat  = {}
            by_agent = {}

            for d in self._history:
                by_cat[d.category]  = by_cat.get(d.category, 0) + 1
                if d.agent_id:
                    by_agent[d.agent_name] = by_agent.get(d.agent_name, 0) + 1

            avg_score = (
                sum(d.score for d in self._history if d.agent_id) / max(1, total - unrouted)
            )

            return {
                "total_routed":    total,
                "unrouted":        unrouted,
                "by_category":     by_cat,
                "by_agent":        by_agent,
                "avg_score":       round(avg_score, 4),
            }

    def recent_decisions(self, n: int = 20) -> List[dict]:
        with self._lock:
            return [d.to_dict() for d in self._history[-n:]]


# ── Default registry with sample agents ──────────────────────────────────────

def build_default_registry() -> AgentRegistry:
    registry = AgentRegistry()

    registry.register(Agent(
        agent_id     = "agent-001",
        name         = "Alice",
        skill_vector = {"Technical": 0.90, "Billing": 0.10, "Legal": 0.00},
        max_capacity = 6,
    ))
    registry.register(Agent(
        agent_id     = "agent-002",
        name         = "Bob",
        skill_vector = {"Technical": 0.20, "Billing": 0.70, "Legal": 0.10},
        max_capacity = 5,
    ))
    registry.register(Agent(
        agent_id     = "agent-003",
        name         = "Carol",
        skill_vector = {"Technical": 0.10, "Billing": 0.10, "Legal": 0.80},
        max_capacity = 4,
    ))
    registry.register(Agent(
        agent_id     = "agent-004",
        name         = "Dave",
        skill_vector = {"Technical": 0.50, "Billing": 0.30, "Legal": 0.20},
        max_capacity = 8,
    ))
    registry.register(Agent(
        agent_id     = "agent-005",
        name         = "Eve",
        skill_vector = {"Technical": 0.40, "Billing": 0.40, "Legal": 0.20},
        max_capacity = 5,
    ))

    return registry


# Module-level singleton
_registry: AgentRegistry | None = None
_reg_lock = threading.Lock()


def get_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        with _reg_lock:
            if _registry is None:
                _registry = build_default_registry()
    return _registry