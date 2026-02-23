Hackathon Challenge: ”Smart-Support” Ticket Routing Engine
System Design & NLP Track

Scenario - A large SaaS company facing scalability issues due to manual support ticket triage, which has become a bottleneck as ticket volume grows into the thousands per day. The task is to design and build a high-throughput, intelligent routing engine that can automatically classify tickets by category, detect urgency levels, distribute workload efficiently among support agents, and remain stable during system failures or sudden spikes in traffic. The goal is to replace slow, manual processes with an automated, resilient, and scalable system that improves response time, operational efficiency, and overall customer experience.

Milestone 1 - The Minimum Viable Router (MVR)
Goal: Establish a functional end-to-end pipeline.
We developed a Minimum Viable Router (MVR) for an intelligent ticket routing system that automates the initial triage of support tickets. Using FastAPI, we built a REST API that accepts incoming tickets in JSON format. A baseline machine learning classifier implemented with scikit-learn categorizes each ticket into Billing, Technical, or Legal using TF-IDF and Logistic Regression. Urgency is detected through a regex-based heuristic that flags keywords such as “broken” or “ASAP” or “critical”, etc and tickets are inserted into an in-memory priority queue using Python’s heapq, ensuring urgent tickets are processed first.

Milestone 2 - The Intelligent queue 
Goal: Transform the Minimum Viable Router into a scalable, concurrency-safe, production-grade intelligent routing system.

In Milestone 2, we transformed the synchronous Minimum Viable Router into a scalable, production-ready asynchronous system. Instead of processing tickets within the API request cycle, we integrated Redis as a message broker and Celery as a background worker, allowing FastAPI to immediately return a 202 Accepted response while classification occurs asynchronously. This decoupled architecture improves responsiveness, supports higher throughput, and enables concurrent processing without blocking the API.
We replaced heuristic urgency detection with transformer-based NLP models that perform category classification (Billing, Technical, Legal) and generate a continuous urgency score S∈[0,1]S \in [0,1]S∈[0,1]. High-urgency tickets automatically trigger real-time Discord alerts. To ensure reliability, we implemented atomic distributed locking using Redis (SET NX EX) to prevent duplicate concurrent processing, added TTL-based fault tolerance, and introduced a /status/{ticket_id} endpoint for asynchronous result retrieval, creating a robust, scalable intelligent routing engine.

Milestone 3 - The Autonomous Orchestrator 
Goal: Build a self-healing, ”agent-aware” system capable of handling ”Flash-Floods.”
We have implemented semantic deduplication using sentence embeddings to compute cosine similarity between incoming tickets.If the similarity score exceeds 0.9, the system groups tickets as related incidents.When more than 10 similar tickets arrive within 5 minutes, individual alerts are suppressed.Instead, the system automatically creates a single Master Incident to prevent ticket storms and reduce agent overload.

Milestone 4 - Circuit Breaker & Skill based routing
Implemented the Circuit Breaker pattern to protect the system from Transformer model failures and latency spikes.The breaker monitors every call to the ML classifier. If the model responds slower than 500ms or raises an exception 3 consecutive times, the breaker opens and automatically switches to a lightweight fallback — the TF-IDF + Logistic Regression model trained in Milestone 1.After 30 seconds, the breaker enters a HALF-OPEN state and sends one probe request to test if the primary model has recovered. A successful probe closes the breaker and restores normal operation.This means the system never goes down due to a slow or broken ML model — it degrades gracefully and self-heals.
 Implemented a constraint optimisation router that assigns each ticket to the best available human agent based on their skill profile and current workload.
Each agent carries a Skill Vector — a set of proficiency scores across the three ticket categories:Subject to:
Agent must have capacity (current_load < max_capacity)
Agent must have non-zero skill for the required category
Agent must be active
The agent with the highest score is assigned the ticket. If scores are tied within a small margin, the agent with the lowest current load wins — ensuring fair distribution under equal skill.



