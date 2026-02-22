import heapq
import re
import time
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

ticket_queue = []

def classify_ticket(text: str):
    text = text.lower()

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
    
    heapq.heappush(ticket_queue, (priority, payload["received_at"], payload))
    
    return {"status": "queued", "category": category, "priority": "High" if priority == 0 else "Normal"}

@app.get("/next")
async def get_next_ticket():
    if not ticket_queue:
        return {"message": "Queue empty"}
    priority, timestamp, ticket_data = heapq.heappop(ticket_queue)
    return ticket_data
