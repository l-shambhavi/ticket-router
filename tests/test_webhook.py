"""
test_webhook.py
===============
Tests Discord webhook 
"""
import pytest
from unittest.mock import patch, MagicMock


def test_sends_correct_payload_when_url_set():
    with patch("milestone2.webhook.httpx.post") as mock_post, \
         patch("milestone2.webhook.DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/fake"):

        from milestone2.webhook import send_alert
        send_alert("tid-x", 0.95, "Technical")

    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs["json"]
    fields  = payload["embeds"][0]["fields"]

    assert payload["embeds"][0]["title"] == "🚨 High Urgency Ticket"
    assert any(f["value"] == "tid-x"       for f in fields)
    assert any(f["value"] == "Technical"   for f in fields)
    assert any(f["value"] == "0.95"        for f in fields)


def test_skips_when_no_url():
    with patch("milestone2.webhook.httpx.post") as mock_post, \
         patch("milestone2.webhook.DISCORD_WEBHOOK_URL", ""):

        from milestone2.webhook import send_alert
        send_alert("tid-y", 0.99, "Billing")

    mock_post.assert_not_called()


def test_network_failure_is_silent():
    """Webhook crash must never propagate — worker should keep running."""
    with patch("milestone2.webhook.httpx.post", side_effect=Exception("timeout")), \
         patch("milestone2.webhook.DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/fake"):

        from milestone2.webhook import send_alert
        send_alert("tid-z", 0.95, "Legal")   # must NOT raise


def test_urgency_rounded_to_3dp():
    with patch("milestone2.webhook.httpx.post") as mock_post, \
         patch("milestone2.webhook.DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/fake"):

        from milestone2.webhook import send_alert
        send_alert("tid-w", 0.87654, "Technical")

    fields = mock_post.call_args.kwargs["json"]["embeds"][0]["fields"]
    score_field = next(f for f in fields if f["name"] == "Urgency Score")
    assert score_field["value"] == "0.877"
