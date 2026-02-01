# AI Medical Chatbot

A conversational medical assistant that helps answer health-related questions, provide guidance, and assist with medical information using AI. This repository contains the code, configuration, and documentation for running and developing the AI Medical Chatbot.

## Features

- Natural language question answering for medical topics
- Context-aware multi-turn conversations
- Pluggable backend model (local model or API-powered)
- Basic input validation and safety filtering
- Deployable as a web service or local CLI tool

## Tech Stack
- Python
- FastAPI
- LangChain
- OpenAI API
- Sentence Transformers
- Pinecone (VectorDB)
- Streamlit
- Docker
- AWS

## Repository structure

- `app/` - Application source code (API server, web UI, business logic)
- `models/` - Model definitions, weights, or adapters (if included)
- `scripts/` - Utility and deployment scripts
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container definition
- `README.md` - This file

Note: If any of the above paths do not exist in the repository, treat them as suggested organization for future development.

## Quickstart (Local)

1. Clone the repository

   git clone https://github.com/Spandan752/AI-Medical-Chatbot.git
   cd AI-Medical-Chatbot

2. Create and activate a virtual environment (Python 3.8+ recommended)

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate    # Windows (PowerShell: .venv\Scripts\Activate.ps1)

3. Install dependencies

   pip install -r requirements.txt

4. Configure environment variables

   Create a `.env` file in the project root (or set env vars directly). Common variables:

   - MODEL_PROVIDER: e.g. `openai`, `local`, or other supported provider
   - MODEL_NAME: model identifier (if using remote API)
   - API_KEY: API key for remote model provider
   - FLASK_ENV / FASTAPI_ENV: environment (development/production)
   - PORT: port for the web service (default: 8000)

   Example `.env`:

   MODEL_PROVIDER=openai
   MODEL_NAME=gpt-4o-mini
   API_KEY=sk-xxxx
   PORT=8000

5. Run the application

   If the repo contains a web server entrypoint (for example `app.py`, `main.py`, or an ASGI app under `app/`), run:

   python app.py
   # or
   uvicorn app.main:app --reload --port $PORT

   If the project uses a different start command, check the repo for an entrypoint and replace the command above.

## Docker

To build and run with Docker (if a Dockerfile exists):

   docker build -t ai-medical-chatbot:latest .
   docker run -p 8000:8000 --env-file .env ai-medical-chatbot:latest

## Development

- Follow the repository structure above. Add tests under `tests/` and use `pytest` for test execution.
- Use pre-commit hooks and linters (flake8, black, isort) to keep code quality high.
- Store secrets in environment variables or a secrets manager; do not commit API keys.

## Security & Safety

- This project is not medical advice. Ensure you include a prominent disclaimer when exposing the chatbot to users.
- Sanitize user input and rate-limit API calls to your model provider.
- Consider incorporating a professional medical disclaimer and escalation paths for emergency situations.
