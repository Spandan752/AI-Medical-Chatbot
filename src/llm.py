from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings, get_logger
# from dotenv import load_dotenv
# import os

logger = get_logger(__name__)
logger.info(f"Initializing LLM model: {settings.MODEL_NAME}")


# load_dotenv()
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# MODEL_NAME = os.getenv('MODEL_NAME')

model = ChatGoogleGenerativeAI(
    api_key = settings.GOOGLE_API_KEY,
    model = settings.MODEL_NAME,
    temperature= settings.MODEL_TEMPERATURE
)   

logger.info("LLM model initialized successfully")