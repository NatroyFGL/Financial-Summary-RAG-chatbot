from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import ChatPromptTemplate
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import mlflow  
import os      

load_dotenv()

#set experiment name in MLflow
mlflow.set_experiment("rag-pipeline")

#Pipeline config
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K = 5
SCORE_THRESHOLD = 0.2
FAISS_INDEX_PATH = "faiss_index"  # MLOps: local path to persist vector store


loader = DirectoryLoader(
    path="./papers",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True,
    use_multithreading=True
)

docs = loader.load()

MARKDOWN_SEPARATORS = [
    "\n#{1,6}",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS
)

splits = text_splitter.split_documents(docs)

# OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL
)

'''# HuggingFace embeddings (open-source alternative)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")'''

#load FAISS index from disk if it exists, otherwise build and save it
if os.path.exists(FAISS_INDEX_PATH):
    print("Loading FAISS index from disk...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )
    vectorstore.save_local(FAISS_INDEX_PATH)  # MLOps: persist to disk
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}/'")

retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k': TOP_K, 'score_threshold': SCORE_THRESHOLD}
)

template = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\"I don't know based on the provided documents.\"\n"
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
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
        "llm_model": "gpt-4o-mini"
    })

    answer = rag_chain.invoke(question)

    mlflow.log_text(question, "question.txt")   #log the question
    mlflow.log_text(answer, "answer.txt")        #log the answer

print(answer)