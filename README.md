# AI Medical Chatbot

> A production-deployed RAG (Retrieval-Augmented Generation) system that answers medical questions grounded in verified clinical documents — not hallucinations.

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange)](https://langchain.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-purple)](https://pinecone.io)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20ECR-orange?logo=amazon-aws)](https://aws.amazon.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=github-actions)](https://github.com/features/actions)

---

## What This Project Demonstrates

This project goes beyond a simple chatbot. It showcases an **end-to-end ML engineering workflow** including:

- Designing and deploying a **production RAG pipeline** with semantic search over medical PDFs
- Building a **REST API** with FastAPI, complete with request/response schema validation
- Containerizing with **Docker** and deploying to **AWS EC2** via **Amazon ECR**
- Automating the full build-test-deploy cycle with **GitHub Actions CI/CD**
- Handling real-world infrastructure challenges (disk management, port mapping, runner configuration)

---

## System Architecture

```
User Query
    │
    ▼
┌─────────────┐     POST /chat      ┌──────────────────────────────────────────┐
│  Client /   │ ──────────────────► │            FastAPI Server                │
│  Streamlit  │                     │                                          │
└─────────────┘                     │  1. Validate input (Pydantic schema)     │
                                    │  2. Embed query (all-MiniLM-L6-v2)       │
                                    │  3. Retrieve top-k docs from Pinecone    │
                                    │  4. Build prompt with context            │
                                    │  5. Generate answer via Gemini LLM       │
                                    │  6. Return structured JSON response      │
                                    └──────────────────────────────────────────┘
                                                        │
                              ┌─────────────────────────┼─────────────────────┐
                              ▼                         ▼                     ▼
                    ┌──────────────────┐   ┌────────────────────┐   ┌──────────────────┐
                    │  Pinecone Index  │   │  Google Gemini LLM │   │  Sentence        │
                    │  (medical-       │   │  (Generation)      │   │  Transformers    │
                    │   chatbot)       │   └────────────────────┘   │  (Embeddings)    │
                    └──────────────────┘                            └──────────────────┘
```

---

## CI/CD Pipeline

```
Push to main
    │
    ▼
┌──────────────────────────────┐
│   CI: GitHub Actions         │
│   (ubuntu-latest runner)     │
│                              │
│  1. Checkout code            │
│  2. Configure AWS creds      │
│  3. Login to Amazon ECR      │
│  4. docker build             │
│  5. docker push → ECR        │
└──────────────┬───────────────┘
               │ on success
               ▼
┌──────────────────────────────┐
│   CD: Self-hosted EC2 Runner │
│                              │
│  1. docker system prune      │
│  2. Pull latest image        │
│  3. docker run on port 8000  │
└──────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Google Gemini (via `langchain_google_genai`) | Answer generation |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) | Semantic vector encoding |
| **Vector DB** | Pinecone (Serverless, cosine similarity, dim=384) | Document retrieval |
| **RAG Framework** | LangChain (LCEL chain) | Orchestration |
| **API** | FastAPI + Uvicorn | REST endpoints |
| **UI** | Streamlit | Web chat interface |
| **Containerization** | Docker | Reproducible builds |
| **Registry** | Amazon ECR | Docker image storage |
| **Compute** | AWS EC2 (self-hosted runner) | Production deployment |
| **CI/CD** | GitHub Actions | Automated build & deploy |

---

## Project Structure

```
AI-Medical-Chatbot/
├── .github/
│   └── workflows/
│       └── cicd.yaml          # CI/CD pipeline (build → ECR → EC2)
├── data/                      # Source medical PDFs for ingestion
├── research/                  # Notebooks for experimentation
├── src/
│   ├── helper.py              # PDF loading, chunking, embedding utils
│   ├── prompts.py             # System prompt for the medical RAG chain
│   └── llm.py                 # Gemini LLM initialization
├── app.py                     # FastAPI server with /chat endpoint
├── streamlit_app.py           # Streamlit chat UI
├── store_index.py             # One-time script: chunk PDFs → Pinecone
├── Dockerfile                 # Container definition
├── requirements.txt           # Python dependencies
└── setup.py                   # Package setup
```

---

## How It Works

### 1. Data Ingestion (`store_index.py`)
Medical PDFs from the `data/` directory are loaded, cleaned, and split into overlapping chunks. Each chunk is embedded using `all-MiniLM-L6-v2` (384-dimensional vectors) and stored in a Pinecone serverless index with cosine similarity.

### 2. RAG Chain (`app.py`)
On every `/chat` request:
1. The user query is embedded with the same model
2. Pinecone retrieves the top-3 most semantically similar chunks
3. A structured prompt (system prompt + retrieved context + user question) is built via LangChain's LCEL
4. Google Gemini generates a grounded answer
5. The answer is returned as a validated JSON response

### 3. Deployment
The app runs in a Docker container on an AWS EC2 instance. Every push to `main` triggers GitHub Actions to rebuild the image, push it to ECR, and redeploy on EC2 automatically.

---

## Running Locally

### Prerequisites
- Python 3.10+
- Docker (optional)
- Pinecone account + API key
- Google AI Studio API key

### Setup

```bash
git clone https://github.com/Spandan752/AI-Medical-Chatbot.git
cd AI-Medical-Chatbot
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:
```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
```

### Ingest Documents (one-time)

```bash
# Add your medical PDFs to the data/ directory, then:
python store_index.py
```

### Run the API

```bash
python app.py
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

### Run with Docker

```bash
docker build -t ai-medical-chatbot .
docker run -p 8000:8000 \
  -e PINECONE_API_KEY=your_key \
  -e GOOGLE_API_KEY=your_key \
  ai-medical-chatbot
```

---

## API Reference

### `GET /`
Health check.

**Response:**
```json
{ "status": "ok", "message": "Medical chatbot is running" }
```

### `POST /chat`
Ask a medical question.

**Request:**
```json
{ "input": "What are the symptoms of type 2 diabetes?" }
```

**Response:**
```json
{
  "response": "Type 2 diabetes commonly presents with increased thirst, frequent urination, fatigue, blurred vision, and slow-healing sores..."
}
```

---

## Deploying to AWS

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `ECR_REPO` | ECR repository name |
| `PINECONE_API_KEY` | Pinecone API key |
| `GOOGLE_API_KEY` | Google Gemini API key |

### EC2 Setup
1. Launch an EC2 instance (Ubuntu 22.04, t3.medium+ recommended)
2. Install Docker: `sudo apt install docker.io -y`
3. Register a GitHub Actions self-hosted runner on the instance
4. Open port 8000 in the EC2 security group (inbound TCP)

Every push to `main` will automatically build, push, and redeploy.

---

## 🧠 Key Engineering Decisions

**Why RAG over fine-tuning?** Medical knowledge changes; RAG allows updating the knowledge base (adding new PDFs) without retraining. It also keeps answers grounded and reduces hallucination risk — critical in a healthcare context.

**Why Pinecone serverless?** Zero infrastructure management, automatic scaling, and cosine similarity search that works natively with sentence-transformer embeddings.

**Why `all-MiniLM-L6-v2`?** It offers an excellent trade-off between speed and semantic quality for retrieval tasks, runs without a GPU, and produces compact 384-dimensional vectors that keep Pinecone costs low.

**Why FastAPI over Flask?** Async-first, automatic OpenAPI docs, built-in Pydantic validation — production-ready out of the box.

---

## Disclaimer

This chatbot is for **informational purposes only** and does not constitute medical advice. Always consult a qualified healthcare professional for medical decisions.

---

## License

MIT License — see [LICENSE](LICENSE) for details.