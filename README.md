# Smart-Support: Intelligent Ticket Routing Engine

## Team Members

- **Dineshprasath** – 22PC10  
- **L Shambhavi** – 22PC17  
- **Muthu Meenakshi** – 22PC20  
- **Naveen Rajesh** – 22PC22  

---

## Hackathon Challenge  
### “Smart-Support” Ticket Routing Engine

This project implements an intelligent ticket routing engine for a large-scale SaaS provider that receives thousands of support tickets daily.

The goal is to replace manual triage with a scalable, resilient, and intelligent system capable of handling high traffic and ticket storms efficiently.

The system evolves across three milestones, progressively improving scalability, intelligence, and reliability.

---

# Milestone 1: Minimum Viable Router (MVR)

### Objective
Build a functional end-to-end ticket routing pipeline.

###  Implementation

- Developed a **REST API using FastAPI** to accept incoming tickets in JSON format.
- Implemented a **baseline ML classifier using scikit-learn**:
  - TF-IDF Vectorization
  - Logistic Regression
  - Categories: **Billing, Technical, Legal**
- Designed a **regex-based urgency detector**:
  - Flags keywords such as “broken”, “ASAP”, “critical”, etc.
- Used Python’s **heapq** to store tickets in an **in-memory priority queue**, ensuring urgent tickets are processed first.
- Single-threaded execution.

### Outcome
A working minimum viable ticket router capable of automated classification and urgency prioritization.

---

# Milestone 2 — Intelligent Asynchronous Routing Engine
## Objective
- In Milestone 2, we transformed the synchronous Minimum Viable Router into a scalable, production-ready asynchronous system. 
- Instead of processing tickets within the API request cycle, we integrated *Redis* as a message broker and *Celery* as a background worker, allowing FastAPI to immediately return a *202 Accepted* response while classification occurs asynchronously. 
- This decoupled architecture improves responsiveness, supports higher throughput, and enables concurrent processing without blocking the API.

We replaced heuristic urgency detection with *transformer-based NLP models* that perform category classification (Billing, Technical, Legal) and generate a continuous urgency score S ∈ [0, 1]. High-urgency tickets automatically trigger real-time *Discord alerts. To ensure reliability, we implemented **atomic distributed locking* using Redis (SET NX EX) to prevent duplicate concurrent processing, added TTL-based fault tolerance, and introduced a /status/{ticket_id} endpoint for asynchronous result retrieval, creating a robust, scalable intelligent routing engine.

---

# Milestone 3: The Autonomous Orchestrator

### Objective
Build a self-healing, agent-aware system capable of handling ticket storms (Flash Floods).

---

## Semantic Deduplication

To prevent overwhelming agents during outages:

- Implemented **sentence embeddings** to compute cosine similarity between incoming tickets:


similarity = (A · B) / (||A|| ||B||)


- If similarity > **0.9**, tickets are grouped as related incidents.
- If more than **10 similar tickets arrive within 5 minutes**:
- Individual alerts are suppressed.
- A single **Master Incident** is automatically created.

---

## Master Incident Creation

Instead of flooding agents with duplicate alerts:

- Similar tickets are grouped.
- A single incident represents the outage.
- Reduces operational overload.
- Enables intelligent incident management.

---
