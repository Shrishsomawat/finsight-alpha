"""
graph/state.py
───────────────
EDUCATIONAL NOTE: In LangGraph, STATE is the single source of truth.
Every agent node reads from state and writes back to state.
Think of it as a shared whiteboard that all agents can read/write.

The state flows through the graph like this:
  START → fetch_data → retrieve_memory → bull_agent → bear_agent
        → judge → risk → synthesize → evaluate → store → END

Each node receives the FULL state dict and returns a dict of UPDATES.
LangGraph merges updates back into state automatically.

TypedDict ensures type safety — you'll know exactly what each field contains.
"""
from typing import TypedDict, List, Dict, Any, Optional


class FinancialState(TypedDict):
    # ── INPUT ──────────────────────────────────────────────────────────────────
    ticker:          str        # Stock symbol e.g. "AAPL"
    question:        str        # User's analysis question

    # ── DATA LAYER (populated by data_agent) ───────────────────────────────────
    price_data:       Dict[str, Any]   # OHLCV summary + raw records
    technicals:       Dict[str, Any]   # RSI, MACD, Bollinger, SMAs
    fundamentals:     Dict[str, Any]   # P/E, revenue growth, margins
    news_headlines:   List[Dict]       # Recent news articles

    # ── MEMORY LAYER (populated by memory_agent) ───────────────────────────────
    past_analyses:    List[Dict]       # Retrieved from ChromaDB vector store
    memory_context:   str             # Formatted string of past analyses

    # ── DEBATE LAYER (the core unique feature) ─────────────────────────────────
    bull_case:        str              # Bull agent's investment thesis
    bull_confidence:  float            # Bull's self-reported confidence (0-10)
    bear_case:        str              # Bear agent's investment thesis
    bear_confidence:  float            # Bear's self-reported confidence (0-10)

    # ── JUDGE LAYER ────────────────────────────────────────────────────────────
    verdict:          str              # Judge's arbitration narrative
    winner:           str              # "bull" | "bear" | "neutral"
    judge_confidence: float            # Judge's confidence in decision (0-10)

    # ── RISK LAYER ─────────────────────────────────────────────────────────────
    risk_metrics:     Dict[str, Any]   # Volatility, drawdown, Sharpe, beta
    risk_level:       str              # "LOW" | "MEDIUM" | "HIGH"
    risk_commentary:  str              # Agent's narrative on risk

    # ── FINAL OUTPUT ───────────────────────────────────────────────────────────
    recommendation:   str              # "STRONG BUY" | "BUY" | "HOLD" | "SELL" | "STRONG SELL"
    price_target:     Optional[float]  # 12-month price target estimate
    final_summary:    str              # Full synthesis narrative

    # ── EVALUATION LAYER ───────────────────────────────────────────────────────
    reasoning_score:  float            # Quality of reasoning (0-10)
    evidence_score:   float            # Quality of evidence used (0-10)
    overall_score:    float            # Composite quality score (0-10)
    eval_notes:       str             # Evaluator's feedback

    # ── METADATA ───────────────────────────────────────────────────────────────
    analysis_id:      str              # Unique ID for this analysis
    errors:           List[str]        # Any errors encountered (graceful degradation)
    steps_completed:  List[str]        # Which nodes have run (for UI progress)