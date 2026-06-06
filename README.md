# Financial Document RAG System

A retrieval-augmented generation (RAG) pipeline that lets you ask natural language questions over a private collection of PDF documents. Built with LangChain, FAISS, OpenAI, and MLflow.

## How it works

Documents are loaded from the papers/ folder, split into chunks, and embedded into a FAISS vector store. 
When you ask a question, the most relevant chunks are retrieved and passed to GPT-4o-mini, which answers strictly based on the provided context with source citations.

## Setup

Install dependencies:
pip install langchain langchain-community langchain-openai langchain-text-splitters pymupdf faiss-cpu mlflow python-dotenv

Create a .env file in the project root:
OPENAI_API_KEY=your_key_here

Add your PDF files into the papers/ folder.

## Usage

Run the chatbot:
python chatbot.py

You will be prompted to enter a question. The answer will be printed and logged to MLflow.

To view MLflow experiment logs:
mlflow ui

## Project Structure

papers/ — place your PDF documents here

faiss_index/ — auto-generated vector store saved to disk after first run

chatbot.py — main pipeline

.env — your OpenAI API key (not committed to git)

## Notes

The FAISS index is saved locally after the first run so it does not rebuild on every execution. The LLM is constrained to answer only from retrieved document context and will say "I don't know based on the provided documents" if the answer is not found.
