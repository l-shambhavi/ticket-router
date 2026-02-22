
Members:
Dineshprasath - 22PC10
L Shambhavi - 22PC17
Muthu Meenakshi - 22PC20
Naveen Rajesh - 22PC22
Hackathon Challenge: ”Smart-Support” Ticket Routing Engine
This project implements an intelligent ticket routing engine for a large-scale SaaS provider that receives support tickets daily.
The system automates ticket triage by classifying tickets into Billing, Technical, or Legal categories using machine learning, detecting urgency through heuristic and sentiment-based scoring, and prioritizing them in a queue for efficient processing.
It evolves across three milestones: 
Milestone 1: A Minimum Viable Router
Using FastAPI, we built a REST API that accepts incoming tickets in JSON format. A baseline machine learning classifier implemented with scikit-learn categorizes each ticket into Billing, Technical, or Legal using TF-IDF and Logistic Regression. Urgency is detected through a regex-based heuristic that flags keywords such as “broken” or “ASAP” or “critical”, etc and tickets are inserted into an in-memory priority queue using Python’s heapq, ensuring urgent tickets are processed first.
The overall objective is to replace manual triage with a scalable, resilient, and intelligent routing system capable of handling high traffic and ticket storms efficiently.
