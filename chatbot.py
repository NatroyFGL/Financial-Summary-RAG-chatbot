"""Command-line chatbot: ask a question about the PDFs in papers/, print the answer and log the run to MLflow"""
import os

import mlflow

#run from the script's folder so mlflow.db and mlruns/ are created here no matter where it's launched from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import CONFIG, ask  # noqa: E402  (loads the FAISS index and the models)

#store MLflow logs in this folder
mlflow.set_tracking_uri("sqlite:///mlflow.db")

#set experiment name in MLflow
mlflow.set_experiment("rag-pipeline")

question = input('Question: ')

#log params and answer under a single MLflow run
with mlflow.start_run():
    mlflow.log_params(CONFIG)

    result = ask(question)

    mlflow.log_params({
        "route": result["route"],               #semantic route that was picked
        "answer_model": result["answer_model"]  #model that actually answered (shows fallbacks)
    })
    mlflow.log_metric("latency_seconds", result["seconds"])
    mlflow.log_text(question, "question.txt")          #log the question
    mlflow.log_text(result["answer"], "answer.txt")    #log the answer

print(result["answer"])
