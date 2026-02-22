"""
classifier.py
=============
Contains two main functions: classify() and urgency_score().
Both use lazy loading to optimize startup time and resource usage.
With lazy loading, models are only downloaded + loaded when the
first real request comes in. Subsequent requests reuse the cached globals.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CATEGORIES = ["Billing", "Technical", "Legal"]

# Globals — None until first request hits classify() or urgency_score()
_cat_tok = None
_cat_mod = None
_sen_tok = None
_sen_mod = None


def _load():
    """Load both models once, reuse forever."""
    global _cat_tok, _cat_mod, _sen_tok, _sen_mod

    if _cat_mod is not None and _sen_mod is not None:
        return  # already loaded

    try:
        if _cat_mod is None:
            print("[classifier] Loading category model...")
            _cat_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            _cat_mod = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=3
            )

        if _sen_mod is None:
            print("[classifier] Loading sentiment model...")
            _sen_tok = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            _sen_mod = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )

        print("[classifier] Models ready.")

    except Exception as e:
        print("[classifier] ERROR loading models:", e)
        _cat_mod = None
        _sen_mod = None


def classify(text: str) -> str:
    """Returns one of: Billing, Technical, Legal"""
    _load()
    if _cat_mod is None:
        print("[classifier] WARNING: category model not loaded, returning 'Technical'")
        return "Technical"  # fallback category

    inputs  = _cat_tok(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = _cat_mod(**inputs)
    probs   = F.softmax(outputs.logits, dim=1)
    return CATEGORIES[torch.argmax(probs).item()]


def urgency_score(text: str) -> float:
    """
    Returns S ∈ [0, 1]. 
    Uses SST-2 sentiment model — index 0 = NEGATIVE = high urgency.
    """
    _load()
    if _sen_mod is None:
        print("[classifier] WARNING: sentiment model not loaded, returning 0.5")
        return 0.5  # fallback value

    inputs  = _sen_tok(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = _sen_mod(**inputs)
    probs   = F.softmax(outputs.logits, dim=1)
    return round(float(probs[0][0].detach()), 4)

def warmup():
    """Call once at worker startup to pre-load models before any requests arrive."""
    print("[classifier] Warming up models...")
    classify("warmup")
    urgency_score("warmup")
    print("[classifier] Warmup complete.")