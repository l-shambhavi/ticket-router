import heapq
import re
import time
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# --- 1. The Priority Queue Logic ---
# heapq is a min-heap. We use (priority, timestamp) to ensure 
# high priority (low number) comes first, then FIFO for ties.
ticket_queue = []

# --- 2. The ML Component (Heuristics) ---
def classify_ticket(text: str):
    text = text.lower()
    
    # Urgency Regex Heuristic
    urgency_pattern = r"(broken|asap|urgent|emergency|critical|down|stop)"
    is_urgent = bool(re.search(urgency_pattern, text))
    priority = 0 if is_urgent else 1
    
    # Baseline Keyword Classifier
    if any(k in text for k in ["invoice", "billing", "charge", "refund"]):
        category = "Billing"
    elif any(k in text for k in ["legal", "gdpr", "tos", "privacy"]):
        category = "Legal"
    else:
        category = "Technical" # Default baseline
        
    return category, priority

# --- 3. System Component (Schema & API) ---
class Ticket(BaseModel):
    id: str
    text: str

@app.post("/ingest")
async def ingest_ticket(ticket: Ticket):
    category, priority = classify_ticket(ticket.text)
    
    # Data payload for the queue
    payload = {
        "id": ticket.id,
        "category": category,
        "text": ticket.text,
        "received_at": time.time()
    }
    
    # Store in priority queue
    # heapq format: (priority_level, timestamp, data)
    heapq.heappush(ticket_queue, (priority, payload["received_at"], payload))
    
    return {"status": "queued", "category": category, "priority": "High" if priority == 0 else "Normal"}

@app.get("/next")
async def get_next_ticket():
    if not ticket_queue:
        return {"message": "Queue empty"}
    # Pop the highest priority item
    priority, timestamp, ticket_data = heapq.heappop(ticket_queue)
    return ticket_data