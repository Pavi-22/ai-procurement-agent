# 🧠 AI Procurement Operations Agent

A multi-agent AI system for procurement intelligence that combines **RAG, tool calling, workflow planning, memory systems, and evaluation metrics** using FastAPI.

---

# 🚀 Key Features

## 📚 1. Document Intelligence (RAG)
- Upload PDF, DOCX, TXT, XLSX files
- Automatic:
  - Text extraction
  - Chunking
  - Embedding (SentenceTransformers)
  - Vector storage using FAISS
- Retrieval with metadata (source, page, score)

---

## 🤖 2. Multi-Agent Architecture

### 📄 Document Agent
- Handles RAG pipeline
- Retrieves and generates answers from documents

### 🔍 Research Agent
- Handles external knowledge simulation (LLM-based)

### 🧠 Decision Agent
- Breaks query into steps (workflow planner)
- Selects tools dynamically
- Executes multi-step reasoning

---

## 🔧 3. Tool Calling System
Supports:

- document_query
- document_retrieve
- web_search
- risk_analyzer
- contract_summarizer

Includes:
- Rule-based + LLM-based routing
- Tool registry execution system

---

## 🧠 4. Memory System

- **Short-term memory:** Last 5 conversations (deque)
- **Long-term memory:** SQLite database
- **Semantic memory:** Embedding-based retrieval

---

## 📊 5. Evaluation Framework
Endpoint: `/evaluate`

Metrics:
- Relevancy
- Precision (approx.)
- Recall
- Hallucination rate
- Agent confidence score

---

## ⚙️ 6. Workflow Planning
- LLM-based query decomposition
- Converts query → step-by-step execution plan
- Supports multi-tool execution

---

## 🌊 7. Streaming Support
- `/chat/stream` provides token-by-token streaming response

---

## 📡 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/documents/upload` | Upload and ingest documents |
| `/chat` | Standard chat with agent |
| `/chat/stream` | Streaming response |
| `/agent/run` | Direct agent execution |
| `/evaluate` | System evaluation |
| `/memory` | Conversation memory |
| `/trace` | Execution logs |

---

## 🧩 System Architecture

User Query
↓
Decision Agent
↓
Workflow Planner
↓
Tool Router
↓
| Document | Research | Tools |

↓
FAISS + LLM + Memory
↓
Final Response


---

## 🛠 Tech Stack

- Python 3.10+
- FastAPI
- FAISS
- SentenceTransformers
- Ollama (LLM)
- PyPDF, python-docx, pandas
- SQLite
- asyncio

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload

Open:

http://127.0.0.1:8000/docs
🧪 Example Usage
Upload Document

Use /documents/upload

Ask Question
What are vendor risks in procurement?
Stream Response

Use /chat/stream

🏁 Project Summary

This system demonstrates:

Multi-agent AI orchestration
RAG-based knowledge system
Tool calling architecture
Workflow planning engine
Memory-augmented LLM system
Evaluation framework for AI systems
✅ Status

✔ Functional Requirements Completed
✔ Multi-Agent System Implemented
✔ RAG Pipeline Implemented
✔ Tool Calling Implemented
✔ Evaluation Framework Implemented
✔ Streaming Support Added
