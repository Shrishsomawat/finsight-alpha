"""
tools/financial_tools.py
─────────────────────────
EDUCATIONAL NOTE: This is the "Data Layer" of our multi-agent system.
Good agents need good tools. Here we wrap yfinance to give agents:
  - Price history (OHLCV)
  - Technical indicators (RSI, MACD, Bollinger Bands, SMAs)
  - Fundamental data (P/E, market cap, revenue growth, etc.)
  - Recent news headlines
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import ta  # Technical Analysis library
from config import LOOKBACK_DAYS, TECH_RSI_PERIOD, TECH_MACD_FAST, TECH_MACD_SLOW, TECH_MACD_SIGNAL


def fetch_price_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch OHLCV price history + key market stats via yfinance.
    Returns a dict with both raw DataFrame (for charting) and summary stats.
    """
    try:
        stock = yf.Ticker(ticker)
        end   = datetime.today()
        start = end - timedelta(days=LOOKBACK_DAYS)

        hist = stock.history(start=start, end=end)
        if hist.empty:
            return {"error": f"No price data found for {ticker}"}

        info = stock.info

        # ── Summary stats ──────────────────────────────────────────────────────
        current_price  = round(hist["Close"].iloc[-1], 2)
        price_52w_high = round(hist["Close"].max(), 2)
        price_52w_low  = round(hist["Close"].min(), 2)
        pct_from_high  = round((current_price - price_52w_high) / price_52w_high * 100, 2)
        ytd_return     = round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 2)

        # ── Average daily volume ────────────────────────────────────────────────
        avg_volume     = int(hist["Volume"].tail(20).mean())

        return {
            "ticker":         ticker.upper(),
            "current_price":  current_price,
            "52w_high":        price_52w_high,
            "52w_low":         price_52w_low,
            "pct_from_high":   pct_from_high,
            "ytd_return_pct":  ytd_return,
            "avg_volume_20d":  avg_volume,
            "market_cap":      info.get("marketCap"),
            "sector":          info.get("sector", "N/A"),
            "industry":        info.get("industry", "N/A"),
            "company_name":    info.get("longName", ticker),
            # Raw DataFrame stored as records for JSON serialization
            "history_records": hist.reset_index().tail(252).to_dict(orient="records"),
            "_df":             hist,          # raw pandas df (not serialized)
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def compute_technical_indicators(price_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    EDUCATIONAL NOTE: Technical indicators are mathematical signals derived
    from price/volume history. Agents use these as quantitative inputs.

    RSI   > 70  → overbought (bearish signal)
    RSI   < 30  → oversold   (bullish signal)
    MACD crosses signal from below → bullish momentum
    Price below BB lower band      → potential reversal
    Price > SMA200                 → long-term uptrend
    """
    if "error" in price_data:
        return {}

    df = price_data["_df"].copy()

    try:
        # ── RSI ─────────────────────────────────────────────────────────────────
        rsi_series   = ta.momentum.RSIIndicator(df["Close"], window=TECH_RSI_PERIOD).rsi()
        rsi_now      = round(float(rsi_series.iloc[-1]), 2)

        # ── MACD ─────────────────────────────────────────────────────────────────
        macd_obj     = ta.trend.MACD(df["Close"], window_fast=TECH_MACD_FAST,
                                      window_slow=TECH_MACD_SLOW,
                                      window_sign=TECH_MACD_SIGNAL)
        macd_val     = round(float(macd_obj.macd().iloc[-1]), 4)
        macd_signal  = round(float(macd_obj.macd_signal().iloc[-1]), 4)
        macd_hist    = round(float(macd_obj.macd_diff().iloc[-1]), 4)
        macd_bullish = bool(macd_val > macd_signal)

        # ── Bollinger Bands ──────────────────────────────────────────────────────
        bb           = ta.volatility.BollingerBands(df["Close"], window=20)
        bb_upper     = round(float(bb.bollinger_hband().iloc[-1]), 2)
        bb_lower     = round(float(bb.bollinger_lband().iloc[-1]), 2)
        bb_pct       = round(float(bb.bollinger_pband().iloc[-1]), 4)  # 0=lower, 1=upper

        # ── Moving Averages ──────────────────────────────────────────────────────
        sma50        = round(float(df["Close"].rolling(50).mean().iloc[-1]), 2)
        sma200       = round(float(df["Close"].rolling(200).mean().iloc[-1]), 2)
        price_now    = price_data["current_price"]
        above_sma50  = bool(price_now > sma50)
        above_sma200 = bool(price_now > sma200)
        golden_cross = bool(sma50 > sma200)   # bullish structure

        # ── Volume Trend (20d avg vs 60d avg) ────────────────────────────────────
        vol_20d      = df["Volume"].tail(20).mean()
        vol_60d      = df["Volume"].tail(60).mean()
        vol_trend    = "increasing" if vol_20d > vol_60d * 1.1 else \
                       "decreasing" if vol_20d < vol_60d * 0.9 else "stable"

        return {
            "rsi":            rsi_now,
            "rsi_signal":     "overbought" if rsi_now > 70 else
                               "oversold"   if rsi_now < 30 else "neutral",
            "macd":           macd_val,
            "macd_signal":    macd_signal,
            "macd_histogram": macd_hist,
            "macd_bullish":   macd_bullish,
            "bb_upper":       bb_upper,
            "bb_lower":       bb_lower,
            "bb_position_pct": bb_pct,
            "sma50":          sma50,
            "sma200":         sma200,
            "above_sma50":    above_sma50,
            "above_sma200":   above_sma200,
            "golden_cross":   golden_cross,
            "volume_trend":   vol_trend,
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Pull fundamental/valuation data from yfinance info dict.
    These are the 'story behind the stock' — what agents use for investment thesis.
    """
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio":           info.get("trailingPE"),
            "forward_pe":         info.get("forwardPE"),
            "peg_ratio":          info.get("pegRatio"),
            "price_to_book":      info.get("priceToBook"),
            "price_to_sales":     info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda":       info.get("enterpriseToEbitda"),
            "profit_margin":      info.get("profitMargins"),
            "operating_margin":   info.get("operatingMargins"),
            "revenue_growth":     info.get("revenueGrowth"),
            "earnings_growth":    info.get("earningsGrowth"),
            "roe":                info.get("returnOnEquity"),
            "debt_to_equity":     info.get("debtToEquity"),
            "current_ratio":      info.get("currentRatio"),
            "free_cashflow":      info.get("freeCashflow"),
            "dividend_yield":     info.get("dividendYield"),
            "analyst_target_price": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationKey"),
            "num_analyst_opinions":   info.get("numberOfAnalystOpinions"),
            "short_percent_float":    info.get("shortPercentOfFloat"),
            "beta":                   info.get("beta"),
            "52w_change":             info.get("52WeekChange"),
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_news(ticker: str, max_items: int = 8) -> List[Dict[str, str]]:
    """
    Fetch recent news via yfinance. No API key required.
    Returns a list of {title, publisher, link} dicts.
    """
    try:
        news_raw = yf.Ticker(ticker).news or []
        return [
            {
                "title":     item.get("content", {}).get("title", ""),
                "publisher": item.get("content", {}).get("provider", {}).get("displayName", ""),
                "link":      item.get("content", {}).get("canonicalUrl", {}).get("url", ""),
            }
            for item in news_raw[:max_items]
            if item.get("content", {}).get("title")
        ]
    except Exception as e:
        return [{"title": f"News fetch error: {e}", "publisher": "", "link": ""}]


def compute_risk_metrics(price_data: Dict[str, Any], fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    """
    EDUCATIONAL NOTE: Risk metrics quantify HOW MUCH could go wrong.
    Even a great company can be a risky investment at the wrong price.

    - Volatility:   standard deviation of daily returns × √252 → annualized
    - Max Drawdown: worst peak-to-trough drop in the lookback period
    - Sharpe:       (return - risk_free) / volatility (simplified, RF ≈ 4.5%)
    - Beta:         how much the stock moves vs the market
    """
    if "error" in price_data or "_df" not in price_data:
        return {"error": "No price data for risk calculation"}

    df = price_data["_df"].copy()
    daily_returns = df["Close"].pct_change().dropna()

    # Annualized volatility
    vol_annual = round(float(daily_returns.std() * np.sqrt(252)), 4)

    # Max drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = round(float(drawdown.min()), 4)

    # Simplified Sharpe (annualized return vs vol, RF = 4.5%)
    ann_return = float(daily_returns.mean() * 252)
    sharpe = round((ann_return - 0.045) / vol_annual, 2) if vol_annual > 0 else None

    # Risk level classification
    beta = fundamentals.get("beta") or 1.0
    if vol_annual > 0.40 or max_dd < -0.35:
        risk_level = "HIGH"
    elif vol_annual > 0.20 or max_dd < -0.20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "annualized_volatility": vol_annual,
        "max_drawdown":          max_dd,
        "sharpe_ratio":          sharpe,
        "beta":                  beta,
        "risk_level":            risk_level,
        "ytd_return":            price_data.get("ytd_return_pct"),
    }