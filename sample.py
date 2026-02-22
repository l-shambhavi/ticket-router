import requests
import time

URL = "http://127.0.0.1:8000/submit"  # replace if your FastAPI runs on a different port

# Sample ticket text — keep it identical to trigger the storm
ticket_text = "System is down in region X. Users cannot login."

# Submit 12 tickets
for i in range(1, 13):
    ticket_id = f"ticket_{i:03d}"
    payload = {"ticket_id": ticket_id, "text": ticket_text}

    response = requests.post(URL, json=payload)
    print(f"Submitted {ticket_id} — status code {response.status_code}")
    
    time.sleep(0.5)  # small delay between tickets (adjust if needed)

print("All tickets submitted. Check Celery worker logs for Master Incident creation.")