import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ncpdp_guide")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", "5"))

PDF_PATH = os.getenv("PDF_PATH", "data/raw/pdf")

PROCESSED_PDF_TEXT_PATH = os.getenv("PROCESSED_PDF_TEXT_PATH", "data/processed/pdf_text.txt")
PROCESSED_TEXT_PATH = os.getenv("PROCESSED_TEXT_PATH", PROCESSED_PDF_TEXT_PATH)
PROCESSED_AUDIO_TEXT_PATH = os.getenv("PROCESSED_AUDIO_TEXT_PATH", "data/processed/audio_text.txt")

LLM_MODEL = os.getenv("LLM_MODEL", "models/gemini-2.5-flash")

# TODO implement REFLECTION_SCORE_THRESHOLD and MAX_ELABORATION_PASSES
# If reflection score are at or below this value, run one elaboration + re-retrieve
REFLECTION_SCORE_THRESHOLD = int(os.getenv("REFLECTION_SCORE_THRESHOLD", "3"))
MAX_ELABORATION_PASSES = int(os.getenv("MAX_ELABORATION_PASSES", "1"))