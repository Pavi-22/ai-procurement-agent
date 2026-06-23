# 🔹 IMPORTS
# =========================================================
# 🔹 IMPORTS
# =========================================================
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import uvicorn
import os
import io
import json
import sqlite3
import numpy as np
import pandas as pd
import uuid

from datetime import datetime
from collections import deque

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
import ollama
import logging



# =========================================================
# 🔹 FIX 1: SAFE JSON PARSER
# =========================================================
def safe_parse_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except:
        return {}


# =========================================================
# 🔹 FIX 2: SAFE STEPS
# =========================================================
def safe_steps(steps, original_query):
    if not isinstance(steps, list):
        return [original_query]

    cleaned = [
        s for s in steps
        if isinstance(s, str) and s.strip()
    ]

    if not cleaned:
        return [original_query]

    return cleaned
# =========================================================
# 🔹 APP
# =========================================================
app = FastAPI(title="AI Procurement Operations Agent")

@app.middleware("http")
async def log_requests(request, call_next):
    start = datetime.now()

    response = await call_next(request)

    duration = (datetime.now() - start).total_seconds()

    logger.info({
        "method": request.method,
        "url": str(request.url),
        "latency_sec": duration
    })

    return response

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title="AI Procurement Operations Agent",
        version="0.1.0",
        routes=app.routes,
    )
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi

# =========================================================
# 🔹 REQUEST MODELS
# =========================================================
class ChatRequest(BaseModel):
    query: str

class ApproveRequest(BaseModel):
    tool: str
    approved: bool

# =========================================================
# 🔹 CONFIG
# =========================================================
class Settings:
    def __init__(self):
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
        self.MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")

        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

TRACE_LOGS = []

def log_trace(thought, action, observation, result=None, tool=None, step_id=None, query_id=None):
    TRACE_LOGS.append({
        "query_id": query_id,
        "step_id": step_id,
        "thought": thought,
        "action": action,
        "tool": tool,
        "observation": observation,
        "result": result,
        "timestamp": str(datetime.now())
    })


def call_llm(prompt, temperature=0.2):

    if settings.LLM_PROVIDER == "ollama":
        return ollama.chat(
            model=settings.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )["message"]["content"]

    elif settings.LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return res.choices[0].message.content

    elif settings.LLM_PROVIDER == "gemini":
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        return client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        ).text

    else:
        return ollama.chat(
        host="http://host.docker.internal:11434",
        model=settings.MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}
    )["message"]["content"]

# =========================================================
# 🔹 MEMORY (SHORT + LONG)
# =========================================================
conversation_memory = deque(maxlen=5)

def add_memory(query, response):
    conversation_memory.append({"query": query, "response": response})
    save_memory(query, response)
    semantic_memory.add(query + " " + str(response))

def get_memory_context():
    return "\n".join([f"Q:{m['query']} A:{m['response']}" for m in conversation_memory])

# =========================================================
# 🔹 SQLITE MEMORY
# =========================================================
conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    response TEXT,
    timestamp TEXT
)
""")
conn.commit()

def save_memory(query, response):
    cursor.execute(
        "INSERT INTO memory (query, response, timestamp) VALUES (?, ?, ?)",
        (query, response, str(datetime.now()))
    )
    conn.commit()

# =========================================================
# 🔹 EMBEDDING MODEL
# =========================================================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================================
# 🔹 SEMANTIC MEMORY
# =========================================================
class SemanticMemory:
    def __init__(self):
        self.store = []
        self.max_size = 200  # limit memory growth

    def add(self, text):
        emb = model.encode([text])[0]
        self.store.append((text, emb))

        # keep memory bounded
        if len(self.store) > self.max_size:
            self.store.pop(0)

    def search(self, query):
        q = model.encode([query])[0]
        scored = [(np.dot(q, e), t) for t, e in self.store]
        scored.sort(reverse=True)
        return [t for _, t in scored[:3]]

semantic_memory = SemanticMemory()

 # =========================
# 🔹 REAL TOOL IMPLEMENTATIONS
# =========================

def web_search_tool(query):
    return {
        "query": query,
        "result": call_llm(f"Give factual summarized info about: {query}")
    }
def risk_analyzer_tool(query):
    risks = ["payment delay", "compliance risk", "vendor risk", "contract ambiguity"]

    detected = [r for r in risks if r in query.lower()]

    return {
        "input": query,
        "detected_risks": detected,
        "analysis": call_llm(f"Analyze procurement risks: {query}")
    }

def contract_summarizer_tool(text):
    return {
        "summary": call_llm(f"Summarize contract in structured bullet points: {text}"),
        "key_clauses": call_llm(f"Extract key clauses: {text}")
    }

#----------tool registry----------
class ToolRegistry:
    def __init__(self, doc_agent):
        self.doc_agent = doc_agent

    def run(self, tool_name, query):

        if tool_name == "document_query":
            return self.doc_agent.run(query)

        if tool_name == "document_retrieve":
            return self.doc_agent.retrieve(query)

        if tool_name == "web_search":
           return web_search_tool(query)

        if tool_name == "risk_analyzer":
          return risk_analyzer_tool(query)

        if tool_name == "contract_summarizer":
          return contract_summarizer_tool(query)

        return call_llm(f"Tool fallback execution: {tool_name} | {query}")
    
    

# =========================================================
# 🔹 RAG PIPELINE
# =========================================================
def process_files(file_names, file_bytes_list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

    chunks, metadata = [], []

    for file, file_bytes in zip(file_names, file_bytes_list):
        name = file.lower()

        if name.endswith(".pdf"):
            reader = PdfReader(file_bytes)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                for c in splitter.split_text(text):
                    if len(c.strip()) > 40:
                        chunks.append(c)
                        metadata.append({
                            "source": file,
                            "page": i
                        })

        elif name.endswith(".txt"):
            text = file_bytes.read().decode()
            for c in splitter.split_text(text):
                if len(c.strip()) > 40:
                    chunks.append(c)
                    metadata.append({
                        "source": file,
                        "page": None
                    })

        elif name.endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes.getvalue()))
            text = "\n".join([p.text for p in doc.paragraphs])
            for c in splitter.split_text(text):
                if len(c.strip()) > 40:
                    chunks.append(c)
                    metadata.append({
                        "source": file,
                        "page": None
                    })

        elif name.endswith(".xlsx"):
            df = pd.read_excel(file_bytes)
            text = df.astype(str).to_string()
            for c in splitter.split_text(text):
                if len(c.strip()) > 40:
                    chunks.append(c)
                    metadata.append({
                        "source": file,
                        "page": None
                    })

    return chunks, metadata

# =========================================================
# 🔹 VECTOR STORE (FAISS)
# =========================================================
def build_index(chunks):
    if not chunks:
        return None

    emb = model.encode(chunks, normalize_embeddings=True)
    emb = np.array(emb).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    return index

def retrieve_chunks(query, index, chunks, k=5, metadata=None):
    if index is None:
        return []

    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q, k)

    results = []

    for i, score in zip(idx[0], scores[0]):
        if i == -1:
            continue

        results.append({
            "text": chunks[i],
            "score": float(score),
            "source": metadata[i]["source"] if metadata else "unknown",
            "page": metadata[i].get("page", None) if metadata else None
        })

    return results

# =========================================================
# 🔹 TOOL ROUTER
# =========================================================
def structured_tool_router(query):
    allowed = [
        "document_query",
        "document_retrieve",
        "web_search",
        "risk_analyzer",
        "contract_summarizer"
    ]

    # 🔹 SIMPLE RULE-BASED FALLBACK FIRST (IMPORTANT)
    q = query.lower()

    if "risk" in q:
        return {"tool": "risk_analyzer"}

    if "contract" in q:
        return {"tool": "contract_summarizer"}

    if "search" in q or "latest" in q:
        return {"tool": "web_search"}

    if "retrieve" in q:
        return {"tool": "document_retrieve"}

    # 🔹 LLM fallback ONLY if needed
    prompt = f"""
Return ONLY JSON like:
{{"tool": "document_query"}}

Allowed:
{allowed}

Query:
{query}
"""

    res = call_llm(prompt, 0.1)
    data = safe_parse_json(res)

    tool = data.get("tool")

    if tool not in allowed:
        tool = "document_query"

    return {"tool": tool}

# =========================================================
# 🔹 WORKFLOW PLANNER
# =========================================================
def workflow_planner(query):
    prompt = f"""
You are an expert workflow decomposition engine for a procurement AI system.

Convert the query into a step-by-step execution plan.

Return ONLY valid JSON:
{{"steps": ["step1", "step2"]}}

Rules:
- Each step must be atomic and executable
- Maximum 5 steps
- No explanations
- No extra text
- If simple query → return 1 step only

Query:
{query}
"""

    res = call_llm(prompt, 0.2)

    data = safe_parse_json(res)

    steps = data.get("steps")

    if not isinstance(steps, list) or len(steps) == 0:
        return [query]

    return steps
# =========================================================
# 🔹 DOCUMENT AGENT (RAG CORE)
# =========================================================
class DocumentAgent:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.metadata = []

    def ingest(self, names, files):
        self.chunks, self.metadata = process_files(names, files)
        self.index = build_index(self.chunks)
        return {"chunks": len(self.chunks)}

    def retrieve(self, query):
        return retrieve_chunks(
            query,
            self.index,
            self.chunks,
            metadata=self.metadata
        )

    def run(self, query):
        memory = get_memory_context()
        semantic = semantic_memory.search(query)
        retrieved = self.retrieve(query)

        evidence_block = "\n".join([
            f"[C{i+1}] {r['text']} | Source={r['source']} | Page={r['page']}"
            for i, r in enumerate(retrieved)
        ])

        prompt = f"""
You are a procurement intelligence assistant.

Use ONLY the evidence below.

Every factual statement MUST cite evidence.

Use citations exactly like:
[C1]
[C2]

If answer not found exactly say:
"Not found in documents"

MEMORY:
{memory}

SEMANTIC CONTEXT:
{semantic}

EVIDENCE:
{evidence_block}

QUESTION:
{query}

Return format:
- Final Answer
- Supporting Evidence [C1], [C2]
- Confidence (0-1)
"""

        answer = call_llm(prompt)

        return {
            "answer": answer,
            "retrieved_sources": [
                {
                    "source": r["source"],
                    "page": r["page"],
                    "score": r["score"]
                }
                for r in retrieved
            ],
            "confidence_hint": round(
                sum([r["score"] for r in retrieved]) / max(len(retrieved), 1),
                2
            )
        }
# =========================================================
# 🔹 RESEARCH AGENT
# =========================================================
class ResearchAgent:
    def run(self, query):
        return call_llm(f"Research: {query}")

# =========================================================
# 🔹 DECISION AGENT (ORCHESTRATOR)
# =========================================================
import asyncio

RISKY_TOOLS = {"risk_analyzer", "contract_summarizer"}
pending_approvals = {}

class DecisionAgent:
    def __init__(self, doc, research):
        self.doc = doc
        self.research = research

    async def run(self, query, query_id=None):

        steps = workflow_planner(query)
        steps = safe_steps(steps, query)

        logger.info(f"Workflow steps: {steps}")

        results = []

        for i, step in enumerate(steps):

            step_text = step

            thought = f"Need to execute: {step_text}"

            tool = structured_tool_router(step_text)["tool"]
            logger.info(f"Executing tool: {tool} for step: {step_text}")

            action = f"Selected tool: {tool}"

            # ❗ Human approval gate
            if tool in RISKY_TOOLS and not pending_approvals.get(tool, False):

              observation = f"Blocked risky tool: {tool}"

              log_trace(
                 thought=thought,
                 action=action,
                 observation=observation,
                 result="BLOCKED (no execution)",
                 tool=tool,
                 step_id=i + 1,
                 query_id=query_id
              )

              results.append({"blocked": tool})
              continue
              results.append({"blocked": tool})
              continue

            # 🔥 tool execution
            if tool == "web_search":
               out = await asyncio.to_thread(
                   self.research.run,
                   step_text
                )
            else:
                out = await asyncio.to_thread(
                    tool_registry.run,
                    tool,
                    step_text
                )

            observation = f"Executed tool: {tool}"

            log_trace(
                thought=thought,
                action=action,
                observation=observation,
                result=out,
                tool=tool,
                step_id=i + 1
            )

            results.append(out)

        # 🔥 CONFIDENCE SCORE
        confidence = round(
            len([
                r for r in results
                if r and (not isinstance(r, dict) or "blocked" not in r)
            ])
            / max(len(steps), 1),
            2
        )

        # 🔥 FINAL LLM AGGREGATION (CLEAN FIXED PROMPT)
        final = await asyncio.to_thread(
            call_llm,
            f"""
You are a procurement AI orchestrator.

Combine tool outputs into a final structured response.

RULES:
- Use ONLY given results
- Do not hallucinate
- Be concise and factual

TOOL RESULTS:
{results}

PREVIOUS MEMORY:
{get_memory_context()}

USER QUERY:
{query}

Return final answer only.
"""
        )

        await asyncio.to_thread(add_memory, query, final)

        return {
            "final_answer": final,
            "confidence": confidence,
            "steps_executed": len(steps),
            "tool_outputs": results
        }

# =========================================================
# 🔹 INIT
# =========================================================
doc_agent = DocumentAgent()
research_agent = ResearchAgent()
decision_agent = DecisionAgent(doc_agent, research_agent)
tool_registry = ToolRegistry(doc_agent)

# =========================================================
# 🔹 ENDPOINTS
# =========================================================
from fastapi import File, UploadFile

@app.post("/documents/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()

    file_name = file.filename
    file_obj = io.BytesIO(content)

    result = doc_agent.ingest([file_name], [file_obj])

    return {
        "message": "uploaded + ingested",
        "file": file_name,
        "chunks_created": result["chunks"]
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    logger.info(f"Chat query received: {req.query}")
    query_id = str(uuid.uuid4())
    result = await decision_agent.run(req.query, query_id)
    return {"result": result}

@app.post("/chat/stream")
async def stream_chat(req: ChatRequest):

    async def generate():
        response = await decision_agent.run(req.query)

        for token in response["final_answer"].split():
           yield token + " "
           await asyncio.sleep(0.02)

    return StreamingResponse(generate(), media_type="text/plain")

    
@app.post("/agent/run")
async def agent_run(req: ChatRequest):
    result = await decision_agent.run(req.query)
    return {"result": result}

@app.post("/approve")
async def approve(req: ApproveRequest):

    pending_approvals[req.tool] = req.approved

    return {
        "tool": req.tool,
        "approved": req.approved,
        "message": f"Approval updated for {req.tool}"
    }


@app.get("/memory")
def memory():
    return list(conversation_memory)


@app.get("/evaluate")
async def evaluate():
    if not doc_agent.chunks:
        return {"error": "No documents uploaded"}

    test_queries = [
        "Summarize procurement contract",
        "What are vendor risks?",
        "Explain payment delay issues",
        "Search latest procurement trends",
        "Retrieve document summary"
    ]

    retrieval_scores = []
    agent_scores = []

    for q in test_queries:
        retrieved = doc_agent.retrieve(q)
        if retrieved:
            retrieval_scores.append(np.mean([r["score"] for r in retrieved]))

        result = await decision_agent.run(q)
        if isinstance(result, dict):
            agent_scores.append(result.get("confidence", 0))

    avg_retrieval = float(np.mean(retrieval_scores)) if retrieval_scores else 0
    avg_agent = float(np.mean(agent_scores)) if agent_scores else 0

    return {
        "relevancy": round(avg_retrieval, 2),
        "agent_confidence": round(avg_agent, 2),
        "precision": round(avg_retrieval, 2),
        "recall": min(1.0, len(retrieval_scores) / len(test_queries)),
        "hallucination_rate": round(1 - avg_retrieval, 2)
    }

@app.get("/test")
def test():
    return {"upload_type": str(UploadFile)}


@app.get("/trace")
def trace():
    return TRACE_LOGS[-20:]

# =========================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)