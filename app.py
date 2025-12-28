from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
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


load_dotenv()

app = FastAPI()

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

embedding_model = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding_model,
    index_name=index_name
    )

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

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

# --------------API Endpoints----------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Medical chatbot is running"}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    try:
        answer = rag_chain.invoke(request.input)
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == '__main__':
   uvicorn.run('app:app', host='0.0.0.0', reload=True, port=8000)