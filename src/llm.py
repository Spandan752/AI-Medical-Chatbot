from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    api_key = GOOGLE_API_KEY,
    model = "gemini-2.5-flash",
    temperature= 1.0
)   