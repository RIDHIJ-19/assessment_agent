#  AI Assessment Recommendation API

An AI-powered system that recommends relevant SHL assessments using a hybrid architecture combining LLM reasoning (Groq), semantic search (FAISS), ONNX embeddings, and rule-based filtering.

---

# 🚀 Live API

Base URL:
https://assessment-agent-n1oq.onrender.com

---

# 📡 API Interface

## /chat (POST)

Request:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Need assessment for entry level Python developer"
    }
  ]
}
```

Response:
```json
{
  "reply": "Got it. Here are 1 assessment that fit Entry-Level with Python.",
  "recommendations": [
    {
      "name": "Python (New)",
      "url": "https://www.shl.com/products/product-catalog/view/python-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": true
}
```

---

## /health (GET)

Response:
```json
{ "status": "ok" }
```

---

# 🧠 System Architecture

User Query → Safety Check → Intent Detection → Constraint Extraction → Semantic Retrieval → Filtering → Ranking → Recommendation Builder

---

# ⚙️ Core Components

- Groq LLM → constraint extraction + ranking
- FAISS → semantic similarity search
- ONNX Runtime → embedding generation
- Rule Engine → filtering + validation + boosting

---

# 🔑 Key Features

- SHL assessment recommendations based on job role
- Skill, level, and test-type aware filtering
- Hybrid LLM + vector search system
- Safe SHL-only domain enforcement
- Structured JSON outputs

---

# 🧩 Tech Stack

- FastAPI
- Python
- FAISS
- ONNX Runtime
- Groq LLM
- Hugging Face Transformers
- NumPy

---

# 🛡️ Reliability & Fallbacks

- LLM retry with exponential backoff
- JSON parsing recovery for malformed outputs
- FAISS fallback to raw catalog search
- Over-filter relaxation when no results found
- Hard SHL-only safety enforcement

---

# 📌 Notes

- System strictly limited to SHL assessment recommendations
- Out-of-scope queries are blocked at safety layer
- Production-grade hybrid AI retrieval system
```
