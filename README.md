# EGEC: LLM Enhanced Explainable Graph for 12345 Citizen Complaints Classification

---
## Dataset

Please refer to https://catalog.data.gov/dataset/consumer-complaint-database to download the Financial Dataset.

---
## Model

### event extraction
We extract events from ticket with few-short icl with Qwen2.5-7B-Instruct.
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
the code is in extract_qwen.py

### Event-ERT 
We construct Event-ERT to peformance ticket routing and hotspot mining,in graph.py
