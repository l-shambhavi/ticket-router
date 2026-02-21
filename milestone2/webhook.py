"""
webhook.py
==========
Tiny module to send Discord alerts via webhook.
Fires a Discord embed when urgency > threshold.
Always silent on failure — never crashes the Celery worker.
"""
import httpx
from milestone2.config import DISCORD_WEBHOOK_URL


def send_alert(ticket_id: str, urgency: float, category: str):
    if not DISCORD_WEBHOOK_URL:
        return
    print("Webhook loaded:", DISCORD_WEBHOOK_URL)
    payload = {
        "username": "Smart Alert Bot",
        "embeds": [{
            "title": "🚨 High Urgency Ticket",
            "color": 16711680,  # red
            "fields": [
                {"name": "Ticket ID",     "value": ticket_id,             "inline": True},
                {"name": "Category",      "value": category,               "inline": True},
                {"name": "Urgency Score", "value": str(round(urgency, 3)), "inline": True},
            ],
            "footer": {"text": "SmartSupport Milestone 2"},
        }]
    }

    try:
        httpx.post(DISCORD_WEBHOOK_URL, json=payload, timeout=3)
    except Exception:
        pass  # never crash the worker
