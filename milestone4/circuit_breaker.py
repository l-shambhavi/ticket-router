"""
circuit_breaker.py
==================
Circuit Breaker pattern for classifier failover.

States:
  CLOSED   → normal operation (Transformer model)
  OPEN     → failover active (lightweight Milestone 1 model)
  HALF_OPEN → probing: try Transformer again after cooldown

If Transformer latency > LATENCY_THRESHOLD_MS (500ms) or raises an exception,
the breaker trips to OPEN and subsequent calls use the fast fallback classifier.

After RECOVERY_TIMEOUT_SEC seconds, the breaker enters HALF_OPEN and probes
once. If the probe succeeds, it resets to CLOSED.
"""

import time
import threading
import re
import pickle
from enum import Enum
from typing import Callable, Tuple

# ── Config ────────────────────────────────────────────────────────────────────
LATENCY_THRESHOLD_MS  = 500   # ms — trip if Transformer is slower than this
FAILURE_THRESHOLD     = 3     # consecutive slow/failed calls before opening
RECOVERY_TIMEOUT_SEC  = 30    # seconds to wait before probing again


# ── State machine ─────────────────────────────────────────────────────────────
class BreakerState(Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Thread-safe circuit breaker that wraps the Transformer classifier.
    Falls back to the Milestone 1 TF-IDF + LogisticRegression model.
    """

    def __init__(
        self,
        latency_threshold_ms: float = LATENCY_THRESHOLD_MS,
        failure_threshold: int       = FAILURE_THRESHOLD,
        recovery_timeout_sec: float  = RECOVERY_TIMEOUT_SEC,
        model_path: str              = "model.pkl",
    ):
        self.latency_threshold_ms = latency_threshold_ms
        self.failure_threshold    = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec

        self._state            = BreakerState.CLOSED
        self._failure_count    = 0
        self._last_failure_ts  = 0.0
        self._lock             = threading.Lock()

        # Metrics
        self.calls_total       = 0
        self.calls_primary     = 0
        self.calls_fallback    = 0
        self.latency_history   = []   # last 100 primary latencies (ms)

        # Load fallback (Milestone 1) model once
        self._fallback_vectorizer = None
        self._fallback_model      = None
        self._load_fallback(model_path)

    # ── Fallback model ────────────────────────────────────────────────────────
    def _load_fallback(self, path: str):
        try:
            with open(path, "rb") as f:
                self._fallback_vectorizer, self._fallback_model = pickle.load(f)
            print(f"[CircuitBreaker] Fallback model loaded from {path}")
        except Exception as e:
            print(f"[CircuitBreaker] WARNING: Could not load fallback model: {e}")

    def _classify_fallback(self, text: str) -> str:
        """Milestone 1 heuristic + ML fallback."""
        # Try sklearn model first
        if self._fallback_model is not None and self._fallback_vectorizer is not None:
            try:
                X = self._fallback_vectorizer.transform([text])
                return self._fallback_model.predict(X)[0]
            except Exception:
                pass

        # Pure heuristic last resort
        text_lower = text.lower()
        if any(k in text_lower for k in ["invoice", "billing", "charge", "refund"]):
            return "Billing"
        if any(k in text_lower for k in ["legal", "gdpr", "tos", "privacy", "contract"]):
            return "Legal"
        return "Technical"

    # ── State helpers ─────────────────────────────────────────────────────────
    @property
    def state(self) -> BreakerState:
        with self._lock:
            if self._state == BreakerState.OPEN:
                elapsed = time.time() - self._last_failure_ts
                if elapsed >= self.recovery_timeout_sec:
                    self._state = BreakerState.HALF_OPEN
                    print("[CircuitBreaker] → HALF_OPEN: probing primary model")
            return self._state

    def _record_success(self, latency_ms: float):
        with self._lock:
            self._failure_count = 0
            self._state         = BreakerState.CLOSED
            self.latency_history.append(latency_ms)
            if len(self.latency_history) > 100:
                self.latency_history.pop(0)

    def _record_failure(self):
        with self._lock:
            self._failure_count   += 1
            self._last_failure_ts  = time.time()
            if self._failure_count >= self.failure_threshold:
                if self._state != BreakerState.OPEN:
                    print(
                        f"[CircuitBreaker] → OPEN after {self._failure_count} "
                        f"consecutive failures/slow calls. "
                        f"Failover to Milestone 1 model for {self.recovery_timeout_sec}s."
                    )
                self._state = BreakerState.OPEN

    # ── Main call ─────────────────────────────────────────────────────────────
    def classify(self, text: str, primary_fn: Callable[[str], str]) -> Tuple[str, str]:
        """
        Classify `text` using primary_fn (Transformer) or fallback.

        Returns:
            (category: str, model_used: str)  where model_used is 'primary' | 'fallback'
        """
        self.calls_total += 1
        current_state = self.state  # may transition OPEN→HALF_OPEN

        if current_state == BreakerState.OPEN:
            # Circuit is open — use fallback immediately
            self.calls_fallback += 1
            category = self._classify_fallback(text)
            print(f"[CircuitBreaker] OPEN — using fallback → {category}")
            return category, "fallback"

        # CLOSED or HALF_OPEN → try primary
        try:
            t0         = time.perf_counter()
            category   = primary_fn(text)
            latency_ms = (time.perf_counter() - t0) * 1000

            if latency_ms > self.latency_threshold_ms:
                print(
                    f"[CircuitBreaker] Latency {latency_ms:.1f}ms > {self.latency_threshold_ms}ms "
                    f"— recording failure"
                )
                self._record_failure()
                # Still return the slow result this time (already computed)
                self.calls_primary += 1
                self.latency_history.append(latency_ms)
                return category, "primary_slow"

            self._record_success(latency_ms)
            self.calls_primary += 1
            return category, "primary"

        except Exception as exc:
            print(f"[CircuitBreaker] Primary model raised: {exc}")
            self._record_failure()
            self.calls_fallback += 1
            category = self._classify_fallback(text)
            return category, "fallback"

    # ── Stats ─────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        avg_latency = (
            sum(self.latency_history) / len(self.latency_history)
            if self.latency_history else 0.0
        )
        return {
            "state":            self.state.value,
            "failure_count":    self._failure_count,
            "calls_total":      self.calls_total,
            "calls_primary":    self.calls_primary,
            "calls_fallback":   self.calls_fallback,
            "avg_latency_ms":   round(avg_latency, 2),
            "p95_latency_ms":   round(
                sorted(self.latency_history)[int(len(self.latency_history) * 0.95)]
                if len(self.latency_history) >= 20 else 0.0, 2
            ),
        }


# ── Module-level singleton (shared by worker + API) ──────────────────────────
_breaker: CircuitBreaker | None = None
_breaker_lock = threading.Lock()


def get_breaker(model_path: str = "model.pkl") -> CircuitBreaker:
    global _breaker
    if _breaker is None:
        with _breaker_lock:
            if _breaker is None:
                _breaker = CircuitBreaker(model_path=model_path)
    return _breaker