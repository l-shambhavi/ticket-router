import heapq
import pickle
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
import re

with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

app = FastAPI()

ticket_queue = []

class Ticket(BaseModel):
    user_id: str
    message: str

def detect_urgency(text):
    urgent_keywords = r"\b(broken|asap|urgent|immediately|down|not working)\b"
    return bool(re.search(urgent_keywords, text.lower()))

def classify_ticket(text):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

@app.post("/submit_ticket")
def submit_ticket(ticket: Ticket):
    
    ticket_id = str(uuid.uuid4())

    category = classify_ticket(ticket.message)
    is_urgent = detect_urgency(ticket.message)

    priority = 1 if is_urgent else 2

    ticket_data = {
        "ticket_id": ticket_id,
        "user_id": ticket.user_id,
        "message": ticket.message,
        "category": category,
        "urgent": is_urgent
    }

    heapq.heappush(ticket_queue, (priority, ticket_data))

    return {
        "status": "Ticket received",
        "ticket_id": ticket_id,
        "category": category,
        "urgent": is_urgent,
        "priority": priority
    }

@app.get("/next_ticket")
def get_next_ticket():
    if not ticket_queue:
        return {"message": "No tickets in queue"}

    priority, ticket = heapq.heappop(ticket_queue)
    return ticket
