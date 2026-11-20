"""
Web demo for the Financial Document RAG chatbot.

FastAPI serves the chat page (frontend/) and a JSON API that runs the same pipeline as chatbot.py.
Deploys to Vercel with zero configuration; run it locally with:  uvicorn app:app --reload
"""
from collections import deque
import os
import threading
import time
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MAX_QUESTION_CHARS = 500
PER_MINUTE = 6      # questions per visitor per minute
PER_DAY = 60        # questions per visitor per day
GLOBAL_DAY = 600    # questions per server instance per day, so a public link can't run up the OpenAI bill

app = FastAPI(
    title="Financial Summary RAG",
    description="Ask questions about financial PDFs; answers are grounded in the documents with page citations.",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)


class RateLimiter:
    """In-memory limits per server instance: enough to stop a public demo being spammed"""

    def __init__(self):
        self.lock = threading.Lock()
        self.day = None
        self.recent = {}  # ip -> timestamps of questions in the last minute
        self.daily = {}   # ip -> questions today
        self.total = 0

    def check(self, ip):
        """Return an error message if this visitor should wait, otherwise record the question"""
        now = time.time()
        with self.lock:
            today = time.strftime("%Y-%m-%d", time.gmtime(now))
            if today != self.day:
                self.day, self.recent, self.daily, self.total = today, {}, {}, 0
            if self.total >= GLOBAL_DAY:
                return "The demo has reached its question limit for today. Please try again tomorrow."
            if self.daily.get(ip, 0) >= PER_DAY:
                return "You've reached today's question limit for this demo. Please try again tomorrow."
            window = self.recent.setdefault(ip, deque())
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= PER_MINUTE:
                return "That's a lot of questions in one minute. Wait a moment, then ask again."
            window.append(now)
            self.daily[ip] = self.daily.get(ip, 0) + 1
            self.total += 1
            return None


limiter = RateLimiter()
_pipeline = None
_pipeline_lock = threading.Lock()


def client_ip(request: Request):
    forwarded = request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def friendly_error(exc: Exception):
    """Turn provider errors into a message that's safe to show visitors (details go to the server log)"""
    name = type(exc).__name__
    if name == "AuthenticationError":
        return "The server's OpenAI API key was rejected. Check OPENAI_API_KEY in the deployment settings."
    if name == "RateLimitError":
        return "OpenAI is rate-limiting the demo's API key right now (or its credit has run out). Try again shortly."
    if name in ("APITimeoutError", "APIConnectionError", "TimeoutError"):
        return "OpenAI didn't respond in time. Please try again."
    return "Something went wrong while answering. Please try again."


def get_pipeline():
    """Import the RAG pipeline on first use: loads LangChain, the FAISS index and the persona embeddings"""
    global _pipeline
    if _pipeline is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(503, "The demo isn't set up yet: the server has no OPENAI_API_KEY.")
        with _pipeline_lock:
            if _pipeline is None:
                try:
                    import rag_pipeline
                except Exception as exc:
                    traceback.print_exc()
                    raise HTTPException(503, friendly_error(exc)) from exc
                _pipeline = rag_pipeline
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)


@app.get("/api/health")
def health():
    """Reports whether the demo can answer, and warms up the pipeline so the first question is quicker"""
    status = {"ok": False, "key_configured": bool(os.getenv("OPENAI_API_KEY"))}
    try:
        rag = get_pipeline()
    except HTTPException as exc:
        status["error"] = exc.detail
        return status
    status.update(
        ok=True,
        index=rag.index_summary(),
        models={"embeddings": rag.EMBEDDING_MODEL, "metrics": rag.LLM_MODEL, "summary": rag.HF_MODEL},
        hf_token_configured=bool(os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")),
    )
    return status


@app.post("/api/ask")
def ask(body: AskRequest, request: Request):
    """Run the RAG pipeline on one question and return the answer with its retrieval trace"""
    question = " ".join(body.question.split())
    if not question:
        raise HTTPException(400, "Type a question first.")
    limit_message = limiter.check(client_ip(request))
    if limit_message:
        raise HTTPException(429, limit_message)
    rag = get_pipeline()
    try:
        return rag.ask(question)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(502, friendly_error(exc)) from exc


# The chat page. API routes above always take priority; on Vercel these files are served from the CDN.
app.frontend("/", directory=os.path.join(BASE_DIR, "frontend"))
