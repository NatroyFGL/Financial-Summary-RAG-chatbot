from operator import itemgetter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.load import dumps, loads
from langchain_core.documents import Document
from langchain_community.utils.math import cosine_similarity
from dotenv import load_dotenv
from langchain_core._api import LangChainBetaWarning
import mlflow
import os
import warnings

warnings.filterwarnings("ignore", category=LangChainBetaWarning)  # dumps/loads used by RAG-Fusion are "beta"

#run from the script's folder so papers/, faiss_index/ and mlruns/ resolve no matter where it's launched from
os.chdir(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

#store MLflow logs in this folder
mlflow.set_tracking_uri("sqlite:///mlflow.db")

#set experiment name in MLflow
mlflow.set_experiment("rag-pipeline")

#Pipeline config
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # LLM router: open-source model for SUMMARY questions
TOP_K = 5
SCORE_THRESHOLD = 0.2
NUM_QUERIES = 4  # RAG-Fusion: number of generated search queries
FAISS_INDEX_PATH = "faiss_index"  # MLOps: local path to persist vector store

# FAISS returns squared L2 distance; OpenAI embeddings are unit length, so cosine similarity = 1 - d/2
def cosine_relevance(distance):
    return 1 - distance / 2

# OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

'''HuggingFace embeddings (open-source alternative)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")'''

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0
)

# Open-source model via Hugging Face Inference API (needs HUGGINGFACEHUB_API_TOKEN in .env)
hf_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id=HF_MODEL,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.01
))


# INDEXING

#load FAISS index from disk if it exists, otherwise do indexing(load PDFs, split, embed and save it)
if os.path.exists(FAISS_INDEX_PATH):
    print("Loading FAISS index from disk...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
        relevance_score_fn=cosine_relevance
    )
else:
    print("Building FAISS index...")
    loader = DirectoryLoader(
        path="./papers",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        strip_whitespace=True,
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        relevance_score_fn=cosine_relevance
    )
    vectorstore.save_local(FAISS_INDEX_PATH)  # MLOps: persist to disk
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}/'")

retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': TOP_K, 'score_threshold': SCORE_THRESHOLD}
)


# RETRIEVAL (RAG-Fusion)

# Generate several search queries from the user's question
template = f"""You are a helpful assistant that generates multiple search queries based on a single input query about financial documents.
Generate multiple search queries related to: {{question}}
Output ({NUM_QUERIES} queries, one per line, no numbering):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion
    | llm
    | StrOutputParser()
    | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
)

def reciprocal_rank_fusion(results: list[list], k=60):
    """Fuse the ranked lists from each query: every doc scores 1 / (rank + k) per list it appears in"""
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)

    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [loads(doc, allowed_objects=[Document]) for doc, _ in reranked[:TOP_K]]

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

def format_docs(docs):
    # Tag each chunk with source:page so the LLM can cite it
    return "\n\n".join(
        f"[{os.path.basename(d.metadata.get('source', 'unknown'))}:{d.metadata.get('page', 0) + 1}]\n{d.page_content}"
        for d in docs
    )


# GENERATION (semantic routing + LLM router)

RULES = (
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) Cite sources by copying the exact [file:page] tag shown above the chunk you used, in parentheses.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

# Each route: a persona (embedded for routing) + the shared strict rules
metrics_persona = (
    "You are a precise financial analyst. You answer questions about specific numbers: "
    "revenue, profit, margins, EPS, cash flow, debt, ratios, growth rates and year-over-year changes. "
    "Quote exact figures with their units and periods, and show any calculation step by step.\n"
)

summary_persona = (
    "You are an experienced financial advisor who explains documents clearly. You answer questions about "
    "overall performance, strategy, risks, outlook, management commentary and key takeaways. "
    "Give a concise, well-structured summary in plain language.\n"
)

personas = {"METRICS": metrics_persona, "SUMMARY": summary_persona}
persona_embeddings = embeddings.embed_documents(list(personas.values()))

# LLM router: each route also picks a model. Numbers need OpenAI's accuracy; summaries go to the
# open-source model, falling back to OpenAI if Hugging Face is down, rate-limited or has no token
model_routes = {
    "METRICS": llm,
    "SUMMARY": hf_llm.with_fallbacks([llm]),
}

def router(input):
    # Pick the persona whose description is most similar to the question, then its prompt + model
    query_embedding = embeddings.embed_query(input["question"])
    similarity = cosine_similarity([query_embedding], persona_embeddings)[0]
    route = list(personas)[similarity.argmax()]
    print(f"Using {route}")
    
    if mlflow.active_run():
        mlflow.log_param("route", route)
    return ChatPromptTemplate.from_template(personas[route] + RULES) | model_routes[route]

def log_model(message):
    # Record which model actually answered (shows when the fallback kicked in)
    model = message.response_metadata.get("model_name") or message.response_metadata.get("model") or HF_MODEL
    print(f"Answered by {model}")
    if mlflow.active_run():
        mlflow.log_param("answer_model", model)
    return message

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion | format_docs,
     "question": itemgetter("question")}
    | RunnableLambda(router)
    | RunnableLambda(log_model)
    | StrOutputParser()
)


question = input('Question: ')

#log params and answer under a single MLflow run
with mlflow.start_run():
    mlflow.log_params({
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "score_threshold": SCORE_THRESHOLD,
        "num_queries": NUM_QUERIES,
        "llm_model": LLM_MODEL,
        "hf_model": HF_MODEL
    })

    answer = final_rag_chain.invoke({"question": question})

    mlflow.log_text(question, "question.txt")   #log the question
    mlflow.log_text(answer, "answer.txt")        #log the answer

print(answer)
