"""
agents/nodes.py — GROQ VERSION
All LLM calls use Groq (free, fast, 14400 req/day)
Model: llama-3.3-70b-versatile
"""
import os
import re
import uuid
from typing import Dict, Any
from groq import Groq

from config import GROQ_API_KEY
from graph.state import FinancialState
from tools.financial_tools import (
    fetch_price_data,
    compute_technical_indicators,
    fetch_fundamentals,
    fetch_news,
    compute_risk_metrics,
)
from vector_store.store import memory_store

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call Groq LLaMA and return text response."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=1500,
    )
    return response.choices[0].message.content.strip()


def _extract_score(text: str, default: float = 6.0) -> float:
    """Extract a numeric score (0-10) from LLM output."""
    matches = re.findall(r'\b([0-9](?:\.[0-9]+)?|10(?:\.0+)?)\s*(?:/\s*10|out of 10)?', text)
    for m in matches:
        val = float(m)
        if 0 <= val <= 10:
            return val
    return default


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1: DATA NODE — fetches all market data, no LLM needed
# ══════════════════════════════════════════════════════════════════════════════
def data_node(state: FinancialState) -> Dict[str, Any]:
    ticker = state["ticker"].upper().strip()
    errors = list(state.get("errors", []))

    price_data   = fetch_price_data(ticker)
    if "error" in price_data:
        errors.append(f"Price data error: {price_data['error']}")

    technicals   = compute_technical_indicators(price_data) if "error" not in price_data else {}
    fundamentals = fetch_fundamentals(ticker)
    news         = fetch_news(ticker)
    risk_metrics = compute_risk_metrics(price_data, fundamentals)

    return {
        "ticker":          ticker,
        "price_data":      {k: v for k, v in price_data.items() if k != "_df"},
        "technicals":      technicals,
        "fundamentals":    fundamentals,
        "news_headlines":  news,
        "risk_metrics":    risk_metrics,
        "risk_level":      risk_metrics.get("risk_level", "MEDIUM"),
        "errors":          errors,
        "steps_completed": state.get("steps_completed", []) + ["📊 Market Data Fetched"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2: MEMORY NODE — retrieves past analyses from ChromaDB
# ══════════════════════════════════════════════════════════════════════════════
def memory_node(state: FinancialState) -> Dict[str, Any]:
    ticker = state["ticker"]
    past   = memory_store.retrieve_past_analyses(ticker)

    if past:
        formatted = "\n\n---\n".join([
            f"[{p['timestamp']}] {p['ticker']} → {p['recommendation']} "
            f"(Risk: {p['risk_level']}, Quality: {p['quality_score']}/10)\n"
            f"{p['document'][:400]}"
            for p in past
        ])
        context = f"PAST ANALYSES FOR {ticker} (from memory):\n{formatted}"
    else:
        context = f"No past analyses found for {ticker}. This is a fresh analysis."

    return {
        "past_analyses":   past,
        "memory_context":  context,
        "steps_completed": state.get("steps_completed", []) + [f"🧠 Memory Retrieved ({len(past)} past analyses)"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3: BULL AGENT — argues the bullish case
# ══════════════════════════════════════════════════════════════════════════════
def bull_node(state: FinancialState) -> Dict[str, Any]:
    ticker = state["ticker"]
    pd     = state.get("price_data", {})
    tech   = state.get("technicals", {})
    fund   = state.get("fundamentals", {})
    news   = state.get("news_headlines", [])
    memory = state.get("memory_context", "")

    news_text = "\n".join([f"• {n['title']} ({n['publisher']})" for n in news[:5]]) or "No recent news"

    system = """You are a seasoned BULL analyst at a top hedge fund. Your job is to construct
the strongest possible BULLISH investment thesis for a given stock. You are optimistic,
data-driven, and find opportunity where others see risk.

You MUST:
1. Lead with your strongest conviction argument
2. Use specific data points provided
3. Address technicals AND fundamentals
4. Consider news catalysts
5. End with: CONFIDENCE: X/10

Be specific. Be analytical. Be the best bull case possible."""

    user = f"""Construct a BULL CASE for {ticker}.

PRICE DATA:
- Current: ${pd.get('current_price', 'N/A')} | 52W High: ${pd.get('52w_high', 'N/A')} | 52W Low: ${pd.get('52w_low', 'N/A')}
- YTD Return: {pd.get('ytd_return_pct', 'N/A')}% | % From High: {pd.get('pct_from_high', 'N/A')}%
- Market Cap: ${pd.get('market_cap', 'N/A')}

TECHNICALS:
- RSI: {tech.get('rsi', 'N/A')} ({tech.get('rsi_signal', 'N/A')})
- MACD Bullish: {tech.get('macd_bullish', 'N/A')} | Histogram: {tech.get('macd_histogram', 'N/A')}
- Above SMA50: {tech.get('above_sma50', 'N/A')} | Above SMA200: {tech.get('above_sma200', 'N/A')}
- Golden Cross: {tech.get('golden_cross', 'N/A')} | Volume Trend: {tech.get('volume_trend', 'N/A')}

FUNDAMENTALS:
- P/E: {fund.get('pe_ratio', 'N/A')} | Forward P/E: {fund.get('forward_pe', 'N/A')}
- Revenue Growth: {fund.get('revenue_growth', 'N/A')} | Earnings Growth: {fund.get('earnings_growth', 'N/A')}
- Profit Margin: {fund.get('profit_margin', 'N/A')} | ROE: {fund.get('roe', 'N/A')}
- Debt/Equity: {fund.get('debt_to_equity', 'N/A')} | Analyst Target: ${fund.get('analyst_target_price', 'N/A')}

RECENT NEWS:
{news_text}

{memory}

USER QUESTION: {state.get('question', 'Should I invest in this stock?')}

Write your BULL CASE (3-4 paragraphs), then end with CONFIDENCE: X/10"""

    result = _call_llm(system, user)
    conf   = _extract_score(result.split("CONFIDENCE:")[-1] if "CONFIDENCE:" in result else result)

    return {
        "bull_case":       result,
        "bull_confidence": conf,
        "steps_completed": state.get("steps_completed", []) + [f"🐂 Bull Case Generated (Confidence: {conf}/10)"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4: BEAR AGENT — argues the bearish case
# ══════════════════════════════════════════════════════════════════════════════
def bear_node(state: FinancialState) -> Dict[str, Any]:
    ticker = state["ticker"]
    pd     = state.get("price_data", {})
    tech   = state.get("technicals", {})
    fund   = state.get("fundamentals", {})
    news   = state.get("news_headlines", [])
    memory = state.get("memory_context", "")

    news_text = "\n".join([f"• {n['title']} ({n['publisher']})" for n in news[:5]]) or "No recent news"

    system = """You are a seasoned BEAR analyst and short-seller known for finding overvalued stocks.
Your job is to construct the strongest possible BEARISH investment thesis. You are skeptical,
contrarian, and see risk where others see opportunity.

You MUST:
1. Lead with your strongest risk/concern
2. Use specific data points to support bearish view
3. Challenge bullish assumptions
4. Identify potential catalysts for downside
5. End with: CONFIDENCE: X/10

Be specific. Be analytical. Be the best bear case possible."""

    user = f"""Construct a BEAR CASE for {ticker}.

PRICE DATA:
- Current: ${pd.get('current_price', 'N/A')} | 52W High: ${pd.get('52w_high', 'N/A')} | 52W Low: ${pd.get('52w_low', 'N/A')}
- YTD Return: {pd.get('ytd_return_pct', 'N/A')}% | % From High: {pd.get('pct_from_high', 'N/A')}%
- Sector: {pd.get('sector', 'N/A')} | Industry: {pd.get('industry', 'N/A')}

TECHNICALS:
- RSI: {tech.get('rsi', 'N/A')} ({tech.get('rsi_signal', 'N/A')})
- MACD Bullish: {tech.get('macd_bullish', 'N/A')} | Histogram: {tech.get('macd_histogram', 'N/A')}
- Above SMA50: {tech.get('above_sma50', 'N/A')} | Above SMA200: {tech.get('above_sma200', 'N/A')}
- BB Position: {tech.get('bb_position_pct', 'N/A')} (0=lower band, 1=upper band)

FUNDAMENTALS:
- P/E: {fund.get('pe_ratio', 'N/A')} | PEG: {fund.get('peg_ratio', 'N/A')}
- Price/Book: {fund.get('price_to_book', 'N/A')} | EV/EBITDA: {fund.get('ev_to_ebitda', 'N/A')}
- Short % Float: {fund.get('short_percent_float', 'N/A')} | Debt/Equity: {fund.get('debt_to_equity', 'N/A')}
- Revenue Growth: {fund.get('revenue_growth', 'N/A')} | Profit Margin: {fund.get('profit_margin', 'N/A')}

RISK METRICS:
- Annualized Volatility: {state.get('risk_metrics', {}).get('annualized_volatility', 'N/A')}
- Max Drawdown: {state.get('risk_metrics', {}).get('max_drawdown', 'N/A')}
- Risk Level: {state.get('risk_level', 'N/A')}

RECENT NEWS:
{news_text}

{memory}

USER QUESTION: {state.get('question', 'Should I invest in this stock?')}

Write your BEAR CASE (3-4 paragraphs), then end with CONFIDENCE: X/10"""

    result = _call_llm(system, user)
    conf   = _extract_score(result.split("CONFIDENCE:")[-1] if "CONFIDENCE:" in result else result)

    return {
        "bear_case":       result,
        "bear_confidence": conf,
        "steps_completed": state.get("steps_completed", []) + [f"🐻 Bear Case Generated (Confidence: {conf}/10)"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5: JUDGE AGENT — arbitrates the debate
# ══════════════════════════════════════════════════════════════════════════════
def judge_node(state: FinancialState) -> Dict[str, Any]:
    ticker    = state["ticker"]
    bull_case = state.get("bull_case", "No bull case")
    bear_case = state.get("bear_case", "No bear case")
    bull_conf = state.get("bull_confidence", 5.0)
    bear_conf = state.get("bear_confidence", 5.0)

    system = """You are the JUDGE in a structured investment debate. Evaluate the quality
of reasoning — not whether you are bullish or bearish yourself.

Structure your response as:
JUDGE'S ANALYSIS: [2-3 paragraphs evaluating both cases]
WINNER: [BULL / BEAR / NEUTRAL]
REASONING: [1 paragraph on why that side won]
CONFIDENCE: [X/10]"""

    user = f"""DEBATE FOR: {ticker}

=== BULL CASE (self-confidence: {bull_conf}/10) ===
{bull_case}

=== BEAR CASE (self-confidence: {bear_conf}/10) ===
{bear_case}

Evaluate this debate. Which case is more compelling?
End with WINNER: [BULL/BEAR/NEUTRAL] and CONFIDENCE: X/10"""

    result = _call_llm(system, user)

    winner = "neutral"
    if "WINNER: BULL" in result.upper() or "WINNER: **BULL" in result.upper():
        winner = "bull"
    elif "WINNER: BEAR" in result.upper() or "WINNER: **BEAR" in result.upper():
        winner = "bear"

    conf = _extract_score(result.split("CONFIDENCE:")[-1] if "CONFIDENCE:" in result else result)

    return {
        "verdict":          result,
        "winner":           winner,
        "judge_confidence": conf,
        "steps_completed":  state.get("steps_completed", []) + [f"⚖️ Judge Ruled: {winner.upper()} wins"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 6: SYNTHESIZER — final recommendation + price target
# ══════════════════════════════════════════════════════════════════════════════
def synthesizer_node(state: FinancialState) -> Dict[str, Any]:
    ticker         = state["ticker"]
    winner         = state.get("winner", "neutral")
    verdict        = state.get("verdict", "")
    risk_lvl       = state.get("risk_level", "MEDIUM")
    fund           = state.get("fundamentals", {})
    price          = state.get("price_data", {}).get("current_price", 0)
    analyst_target = fund.get("analyst_target_price")

    system = """You are the CIO delivering the final investment decision.

Your output MUST include:
RECOMMENDATION: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
PRICE TARGET: $X (12-month estimate)
INVESTMENT THESIS: [3-4 paragraphs]
KEY CATALYSTS: [3 bullet points for upside]
KEY RISKS: [3 bullet points for downside]
TIME HORIZON: [Short/Medium/Long term]"""

    user = f"""Deliver the FINAL INVESTMENT DECISION for {ticker}.

DEBATE OUTCOME: {winner.upper()} case won
JUDGE'S VERDICT: {verdict[:600]}

RISK LEVEL: {risk_lvl}
CURRENT PRICE: ${price}
ANALYST CONSENSUS TARGET: ${analyst_target}
ANALYST RECOMMENDATION: {fund.get('analyst_recommendation', 'N/A')}

USER QUESTION: {state.get('question', 'Should I invest?')}

Deliver your final recommendation with price target and complete thesis."""

    result = _call_llm(system, user)

    rec = "HOLD"
    for r in ["STRONG BUY", "STRONG SELL", "BUY", "SELL", "HOLD"]:
        if r in result.upper():
            rec = r
            break

    target = None
    pt_match = re.search(r'PRICE TARGET[:\s]*\$([0-9,]+(?:\.[0-9]+)?)', result, re.IGNORECASE)
    if pt_match:
        try:
            target = float(pt_match.group(1).replace(",", ""))
        except:
            pass

    return {
        "recommendation":  rec,
        "price_target":    target,
        "final_summary":   result,
        "steps_completed": state.get("steps_completed", []) + [f"🎯 Recommendation: {rec}"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 7: EVALUATOR — scores quality of the analysis
# ══════════════════════════════════════════════════════════════════════════════
def evaluator_node(state: FinancialState) -> Dict[str, Any]:
    system = """You are an AI evaluation specialist who scores financial analysis quality.

Format your response EXACTLY as:
REASONING_SCORE: X/10 — [brief reason]
EVIDENCE_SCORE: X/10 — [brief reason]
OVERALL_SCORE: X/10 — [brief reason]
NOTES: [1 paragraph of feedback]"""

    user = f"""Evaluate this financial analysis for {state.get('ticker')}:

BULL CASE: {state.get('bull_case', '')[:400]}
BEAR CASE: {state.get('bear_case', '')[:400]}
JUDGE VERDICT: {state.get('verdict', '')[:400]}
FINAL RECOMMENDATION: {state.get('recommendation')} | RISK: {state.get('risk_level')}

Score on reasoning quality, evidence quality, and overall trustworthiness."""

    result = _call_llm(system, user)

    r_score = _extract_score(result.split("REASONING_SCORE:")[1].split("\n")[0] if "REASONING_SCORE:" in result else "6")
    e_score = _extract_score(result.split("EVIDENCE_SCORE:")[1].split("\n")[0]  if "EVIDENCE_SCORE:"  in result else "6")
    o_score = _extract_score(result.split("OVERALL_SCORE:")[1].split("\n")[0]   if "OVERALL_SCORE:"   in result else "6")
    notes   = result.split("NOTES:")[-1].strip()[:500] if "NOTES:" in result else ""

    return {
        "reasoning_score": r_score,
        "evidence_score":  e_score,
        "overall_score":   o_score,
        "eval_notes":      notes,
        "steps_completed": state.get("steps_completed", []) + [f"📋 Quality Score: {o_score}/10"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 8: STORE NODE — persists to ChromaDB vector store
# ══════════════════════════════════════════════════════════════════════════════
def store_node(state: FinancialState) -> Dict[str, Any]:
    try:
        analysis_id = memory_store.store_analysis(
            ticker         = state["ticker"],
            recommendation = state.get("recommendation", "HOLD"),
            summary        = state.get("final_summary", ""),
            bull_case      = state.get("bull_case", ""),
            bear_case      = state.get("bear_case", ""),
            verdict        = state.get("verdict", ""),
            risk_level     = state.get("risk_level", "MEDIUM"),
            quality_score  = state.get("overall_score", 0.0),
            price_at_time  = state.get("price_data", {}).get("current_price"),
        )
    except Exception as e:
        analysis_id = f"store_error_{uuid.uuid4().hex[:6]}"

    return {
        "analysis_id":     analysis_id,
        "steps_completed": state.get("steps_completed", []) + ["💾 Analysis Saved to Memory"],
    }