from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

from config import settings, get_logger

import os
from dotenv import load_dotenv

from src.helper import download_embeddings
from src.prompts import system_prompt
from src.llm import model

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = get_logger(__name__)


load_dotenv()

app = FastAPI(
   title=settings.API_TITLE,
   version=settings.API_VERSION,
   debug=settings.DEBUG
)

# ----------Schemas-----------
# Request Schema
class QueryRequest(BaseModel):
  input: str

# Response Schema
class QueryResponse(BaseModel):
  response: str

# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -----------Vector Store and RAG Chain Setup------------
logger.info("Initializing embedding model")
embedding_model = download_embeddings()

# index_name = "medical-chatbot"

logger.info("Connecting to Pinecone")
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding_model,
    index_name=settings.PINECONE_INDEX_NAME
    )


retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": settings.TOP_K_RETRIEVAL}
)


model = model

prompt = ChatPromptTemplate.from_messages([
   ("system", system_prompt),
   ("human", "{input}")
])


rag_chain = (
    { "context": retriever, "input": RunnablePassthrough() }
    | prompt
    | model
    | StrOutputParser()
)

logger.info("RAG Chain initialized successfully")

# --------------API Endpoints----------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Medical chatbot is running", "environment": settings.ENVIRONMENT}

logger.info("Health check completed successfully")

    

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    """Chat endpoint for medical queries."""
    try:
        logger.info(f"Processing query: {request.input[:50]}...")
        
        if not request.input.strip():
            logger.warning("Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        answer = rag_chain.invoke(request.input)
        logger.info("Query processed successfully")
        return {"response": answer}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == '__main__':
    logger.info(f"Starting Medical Chatbot on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        'app:app',
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )