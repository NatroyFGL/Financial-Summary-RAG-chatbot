# Financial Document RAG System

A retrieval-augmented generation (RAG) pipeline that lets you ask natural language questions over a private collection of PDF documents. Built with LangChain, FAISS, OpenAI, and MLflow.

# Project Structure

papers/ — place your PDF documents here

faiss_index/ — auto-generated vector store saved to disk after first run

chatbot.py — main pipeline

.env — your OpenAI API key (not committed to git)

# How this repo works

Documents are loaded from the papers/ folder, split into chunks, and embedded into a FAISS vector store. 
When you ask a question, the most relevant chunks are retrieved and passed to GPT-4o-mini, which answers strictly based on the provided context with source citations.

# Pipeline

1) Indexing: PDFs are split into chunks, embedded with text-embedding-3-large and stored in a FAISS index saved to disk.
2) Retrieval (RAG-Fusion): the LLM rewrites the question into 4 search queries; results are merged with Reciprocal Rank Fusion.
3) Semantic routing: the question is embedded and matched to the closest persona: METRICS (exact figures, calculations) or SUMMARY (plain-language overview).
4) LLM router: each route also picks a model:
- Route METRICS: gpt-4o-mini (more reliable with exact figures and citation)
- Route SUMMARY: Qwen/Qwen2.5-7b-Instruct (open source model for narrative answers)

The Hugging Face route falls back to OpenAI automatically if the API is down, rate-limited or no token is set.
5) MLOps: each run logs its config, chosen route, the model that actually answered, and the Q&A to MLflow.


# Setup

Install dependencies:
pip install langchain langchain-community langchain-openai langchain-huggingface langchain-text-splitters pymupdf faiss-cpu mlflow python-dotenv

Create a .env file in the project root:
OPENAI_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here  # optional, enables the open-source model route

Add your PDF files into the papers/ folder.

# Usage

Run the chatbot:
python chatbot.py

You will be prompted to enter a question. The answer will be printed and logged to MLflow.

To view MLflow experiment logs:
mlflow ui


# Notes

The FAISS index is saved locally after the first run so it does not rebuild on every execution. The LLM is constrained to answer only from retrieved document context and will say "I don't know based on the provided documents" if the answer is not found.









