import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ncpdp_guide")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", "5"))

PDF_PATH = "data/raw/eis_transaction_guide_ncpdp_v4.2.pdf"
PROCESSED_TEXT_PATH = "data/processed/pdf_text.txt"

PROCESSED_PDF_TEXT_PATH = os.getenv("PROCESSED_PDF_TEXT_PATH", "data/processed/pdf_text.txt")
PROCESSED_AUDIO_TEXT_PATH = os.getenv("PROCESSED_AUDIO_TEXT_PATH", "data/processed/audio_text.txt")

LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-2.5-flash")