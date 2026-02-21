"""
test_classifier.py
==================
Tests classify() and urgency_score() without downloading any models.
We patch _load() so it does nothing, then inject mock model/tokenizer objects.
"""
import torch
import pytest
from unittest.mock import MagicMock, patch
import milestone2.classifier as clf


def fake_model_output(logits: list):
    """Helper — returns a mock that looks like a HuggingFace model output."""
    mock_out        = MagicMock()
    mock_out.logits = torch.tensor([logits])
    return mock_out


def fake_tokenizer(text, **kwargs):
    """Returns a minimal dict that the model accepts."""
    return {"input_ids": torch.zeros(1, 5, dtype=torch.long)}


# ── classify() ────────────────────────────────────────────────────────────────
@patch.object(clf, "_load")
def test_classify_billing(mock_load):
    clf._cat_tok = fake_tokenizer
    clf._cat_mod = MagicMock(return_value=fake_model_output([10.0, 0.1, 0.1]))  # Billing wins
    assert clf.classify("My invoice is wrong") == "Billing"


@patch.object(clf, "_load")
def test_classify_technical(mock_load):
    clf._cat_tok = fake_tokenizer
    clf._cat_mod = MagicMock(return_value=fake_model_output([0.1, 10.0, 0.1]))  # Technical wins
    assert clf.classify("API keeps throwing 500 errors") == "Technical"


@patch.object(clf, "_load")
def test_classify_legal(mock_load):
    clf._cat_tok = fake_tokenizer
    clf._cat_mod = MagicMock(return_value=fake_model_output([0.1, 0.1, 10.0]))  # Legal wins
    assert clf.classify("GDPR compliance issue") == "Legal"


@patch.object(clf, "_load")
def test_classify_returns_valid_category(mock_load):
    clf._cat_tok = fake_tokenizer
    clf._cat_mod = MagicMock(return_value=fake_model_output([5.0, 3.0, 1.0]))
    result = clf.classify("Some ticket text")
    assert result in {"Billing", "Technical", "Legal"}


# ── urgency_score() ───────────────────────────────────────────────────────────
@patch.object(clf, "_load")
def test_urgency_score_is_float(mock_load):
    clf._sen_tok = fake_tokenizer
    clf._sen_mod = MagicMock(return_value=fake_model_output([0.8, 0.2]))
    score = clf.urgency_score("System is down!")
    assert isinstance(score, float)


@patch.object(clf, "_load")
def test_urgency_score_in_range(mock_load):
    clf._sen_tok = fake_tokenizer
    clf._sen_mod = MagicMock(return_value=fake_model_output([0.7, 0.3]))
    score = clf.urgency_score("ASAP fix needed")
    assert 0.0 <= score <= 1.0


@patch.object(clf, "_load")
def test_high_urgency_for_negative_text(mock_load):
    """Dominant NEGATIVE logit (index 0) should produce score > 0.5."""
    clf._sen_tok = fake_tokenizer
    clf._sen_mod = MagicMock(return_value=fake_model_output([5.0, 0.1]))  # strongly NEGATIVE
    score = clf.urgency_score("Critical outage — everything is broken ASAP!")
    assert score > 0.5, f"Expected > 0.5, got {score}"


@patch.object(clf, "_load")
def test_low_urgency_for_positive_text(mock_load):
    """Dominant POSITIVE logit (index 1) should produce score < 0.5."""
    clf._sen_tok = fake_tokenizer
    clf._sen_mod = MagicMock(return_value=fake_model_output([0.1, 5.0]))  # strongly POSITIVE
    score = clf.urgency_score("Everything is working great, thank you!")
    assert score < 0.5, f"Expected < 0.5, got {score}"
