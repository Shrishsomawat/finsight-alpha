import os
from dotenv import load_dotenv

load_dotenv()

# ── Groq (Free) ─────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL        = "llama-3.3-70b-versatile"
MAX_TOKENS   = 1500

# ── Vector DB ────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION  = "financial_analyses"

# ── Analysis Settings ────────────────────────
LOOKBACK_DAYS    = 365
TECH_RSI_PERIOD  = 14
TECH_MACD_FAST   = 12
TECH_MACD_SLOW   = 26
TECH_MACD_SIGNAL = 9
MEMORY_TOP_K     = 3