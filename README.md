# High-Performance Event-Driven RAG System

A production-grade Retrieval-Augmented Generation (RAG) pipeline optimized for **low latency**, **durable execution**, and **zero-cost embeddings**. This system allows users to ingest complex PDFs and perform semantic queries with near-instant responses.



## 🚀 Key Features
* **Event-Driven Orchestration**: Powered by **Inngest** to handle long-running background tasks (ingestion/embedding) with built-in retries and state management.
* **Blazing Fast Inference**: Leverages **Groq’s LPU hardware** (Llama-3.3-70b) for ultra-low latency text generation.
* **Cost-Efficient Embeddings**: Uses a local **HuggingFace** model (`BAAI/bge-small-en-v1.5`) to vectorize documents for free, ensuring data privacy.
* **Scalable Vector Storage**: Utilizes **Qdrant** running in **Docker** for high-speed similarity search and isolated data management.
* **System Resilience**: Implements intelligent throttling and rate-limiting to manage CPU-intensive tasks.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM (The Brain)** | Groq (Llama-3.3-70b-versatile) |
| **Embeddings** | HuggingFace (`BAAI/bge-small-en-v1.5`) |
| **Orchestration** | Inngest (Event-driven background workers) |
| **Vector Database** | Qdrant (via Docker) |
| **RAG Framework** | LlamaIndex |
| **Backend / UI** | FastAPI / Streamlit |

---

## 📐 Architecture & Logic

The system is split into two primary pipelines to ensure the UI remains responsive while heavy processing happens in the background:

### 1. The Ingestion Pipeline
1.  **Event Trigger**: Streamlit sends an event to the Inngest Dev Server.
2.  **Durable Execution**: Inngest triggers a FastAPI background worker.
3.  **Semantic Chunking**: Documents are parsed and split into 1,000-character chunks with a 200-character overlap.
4.  **Local Vectorization**: Chunks are converted into **384-dimension** vectors locally.
5.  **Upsert**: Vectors and text payloads are saved to the Qdrant container.

### 2. The Retrieval Pipeline
1.  **Query Embedding**: The user's question is vectorized using the local HuggingFace model.
2.  **Top-K Search**: Qdrant performs a **Cosine Similarity** search to find the most relevant context.
3.  **Contextual Generation**: Relevant chunks are sent to Groq with a strict system prompt to generate an answer grounded only in the provided data.

---

## 🏗️ Why This Architecture? (Engineering Decisions)
* **Why Inngest over standard async?** Traditional async tasks in Python can be fragile. Inngest provides **observability** and **idempotency**, ensuring that if an API call fails, the system retries exactly where it left off without duplicating data.
* **Why Local Embeddings?** Moving embeddings from the cloud (OpenAI) to a local model (BAAI) reduced costs to zero and significantly increased privacy for sensitive documents.
* **Why Docker for Qdrant?** Containerization allows for an isolated, persistent storage layer that is easy to deploy and manage independently of the application code.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
* Docker Desktop
* Python 3.13
* Groq API Key

### 2. Environment Configuration (`.env`)
```env
GROQ_API_KEY=------
INNGEST_DEV=1
```

### 3. Execution
Run the "Trio" of services in separate terminals:

* **Database**: `docker start qdrantRagDb`
* **Backend**: `uv run uvicorn main:app --reload`
* **Orchestrator**: `npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest`
* **Frontend**: `uv run streamlit run streamlit_app.py`

---
