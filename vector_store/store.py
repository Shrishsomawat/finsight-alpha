"""
vector_store/store.py
──────────────────────
EDUCATIONAL NOTE: Vector Databases give agents PERSISTENT MEMORY.
Instead of forgetting everything between sessions, the system can:
  1. STORE: Save every completed analysis as an embedding
  2. RETRIEVE: When analyzing a ticker again, pull relevant past analyses
  3. LEARN: The judge and evaluator can see if past recommendations were correct

This is what separates a "stateful AI system" from a one-shot prompt.

How it works:
  - Each analysis is converted to a vector (embedding) via ChromaDB's
    default embedding function (sentence-transformers under the hood)
  - Vectors are stored with metadata: ticker, date, recommendation, score
  - At query time, we find the closest vectors to "ticker X analysis"
  - Retrieved context is injected into the agents' prompts
"""
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import uuid
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, MEMORY_TOP_K


class FinancialMemoryStore:
    """
    A thin wrapper around ChromaDB that provides financial-analysis-specific
    read/write operations. Think of it as the 'long-term memory' of FinSight.
    """

    def __init__(self):
        # PersistentClient ensures data survives between app restarts
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Default embedding function uses sentence-transformers (all-MiniLM-L6-v2)
        # This converts text → 384-dimensional vector for similarity search
        self.ef = embedding_functions.DefaultEmbeddingFunction()

        # Get or create the collection (like a "table" in a relational DB)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=self.ef,
            metadata={"description": "FinSight financial analysis memory"}
        )

    def store_analysis(
        self,
        ticker:         str,
        recommendation: str,    # BUY / SELL / HOLD
        summary:        str,    # Full analysis text (this gets embedded)
        bull_case:      str,
        bear_case:      str,
        verdict:        str,
        risk_level:     str,
        quality_score:  float,
        price_at_time:  Optional[float] = None,
    ) -> str:
        """
        Store a completed analysis in the vector DB.
        Returns the analysis ID for reference.
        """
        analysis_id = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp   = datetime.now().isoformat()

        # The 'document' is what gets embedded — make it rich in financial language
        document = f"""
Ticker: {ticker} | Date: {timestamp[:10]} | Recommendation: {recommendation}
Risk Level: {risk_level} | Quality Score: {quality_score}/10

BULL CASE: {bull_case[:500]}

BEAR CASE: {bear_case[:500]}

JUDGE VERDICT: {verdict[:500]}

FINAL SUMMARY: {summary[:800]}
        """.strip()

        self.collection.add(
            documents=[document],
            metadatas=[{
                "ticker":         ticker.upper(),
                "recommendation": recommendation,
                "risk_level":     risk_level,
                "quality_score":  str(quality_score),
                "price_at_time":  str(price_at_time) if price_at_time else "N/A",
                "timestamp":      timestamp,
                "analysis_id":    analysis_id,
            }],
            ids=[analysis_id]
        )
        return analysis_id

    def retrieve_past_analyses(self, ticker: str, top_k: int = MEMORY_TOP_K) -> List[Dict[str, Any]]:
        """
        EDUCATIONAL NOTE: This is semantic retrieval, not keyword search.
        We query by ticker name + context so ChromaDB finds the most semantically
        relevant past analyses — even if wording differs.

        Returns list of past analysis dicts with metadata.
        """
        total = self.collection.count()
        if total == 0:
            return []

        results = self.collection.query(
            query_texts=[f"Financial analysis for {ticker} stock investment recommendation"],
            n_results=min(top_k, total),
            where={"ticker": ticker.upper()}   # filter to same ticker
        )

        if not results["documents"] or not results["documents"][0]:
            # If no ticker-specific results, try broader query
            results = self.collection.query(
                query_texts=[f"Financial analysis for {ticker} stock investment recommendation"],
                n_results=min(top_k, total),
            )

        past = []
        for doc, meta in zip(
            results["documents"][0] if results["documents"] else [],
            results["metadatas"][0] if results["metadatas"] else []
        ):
            past.append({
                "document":      doc,
                "ticker":        meta.get("ticker"),
                "recommendation":meta.get("recommendation"),
                "risk_level":    meta.get("risk_level"),
                "quality_score": meta.get("quality_score"),
                "timestamp":     meta.get("timestamp", "")[:10],
                "analysis_id":   meta.get("analysis_id"),
            })
        return past

    def get_all_analyses(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent analyses for the memory browser UI."""
        total = self.collection.count()
        if total == 0:
            return []
        results = self.collection.get(limit=min(limit, total), include=["documents", "metadatas"])
        out = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            out.append({**meta, "document": doc[:300] + "..."})
        # Sort by timestamp descending
        out.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return out

    def count(self) -> int:
        return self.collection.count()


# Singleton instance — import this everywhere
memory_store = FinancialMemoryStore()