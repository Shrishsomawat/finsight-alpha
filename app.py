"""
app.py — FinSight Alpha: Multi-Agent Financial Intelligence System
──────────────────────────────────────────────────────────────────
Bloomberg Terminal aesthetic: dark background, green data, red warnings.
Real-time agent execution streaming via LangGraph .stream()
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight Alpha",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Terminal CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg:       #0a0e1a;
    --surface:  #0f1629;
    --surface2: #151d35;
    --border:   #1e2d50;
    --green:    #00d68f;
    --red:      #ff3b5c;
    --yellow:   #ffb400;
    --blue:     #4d9fff;
    --text:     #c8d8f0;
    --muted:    #5a6d8a;
    --accent:   #7c5cfc;
}

/* ── Base ── */
.stApp { background: var(--bg); color: var(--text); font-family: 'IBM Plex Sans', sans-serif; }
.main .block-container { padding: 1rem 2rem; max-width: 1400px; }
h1,h2,h3,h4,h5,h6 { font-family: 'IBM Plex Mono', monospace; color: var(--text); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 4px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,92,252,0.2) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #4d9fff, var(--accent)) !important;
    color: white !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    padding: 0.6rem 2rem !important;
    border-radius: 4px !important;
    width: 100% !important;
    text-transform: uppercase !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.card-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 0.8rem;
}

/* ── Recommendation badge ── */
.rec-badge {
    display: inline-block;
    padding: 0.4rem 1.5rem;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 1.4rem;
    letter-spacing: 0.1em;
}
.rec-STRONG.BUY, .rec-BUY    { background: rgba(0,214,143,0.15); color: var(--green); border: 1px solid var(--green); }
.rec-HOLD                     { background: rgba(255,180,0,0.15);  color: var(--yellow); border: 1px solid var(--yellow); }
.rec-SELL, .rec-STRONG.SELL  { background: rgba(255,59,92,0.15);  color: var(--red);   border: 1px solid var(--red); }

/* ── Bull/Bear cards ── */
.bull-card {
    background: rgba(0,214,143,0.05);
    border: 1px solid rgba(0,214,143,0.3);
    border-radius: 6px; padding: 1rem;
}
.bear-card {
    background: rgba(255,59,92,0.05);
    border: 1px solid rgba(255,59,92,0.3);
    border-radius: 6px; padding: 1rem;
}
.bull-title { color: var(--green); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 0.15em; }
.bear-title { color: var(--red);   font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 0.15em; }

/* ── Metric boxes ── */
.metric-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 0.2rem;
}
.metric-green { color: var(--green) !important; }
.metric-red   { color: var(--red)   !important; }
.metric-yellow { color: var(--yellow) !important; }

/* ── Step tracker ── */
.step-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--green);
    padding: 0.2rem 0;
    opacity: 0;
    animation: fadeIn 0.4s forwards;
}
@keyframes fadeIn { to { opacity: 1; } }

/* ── Score bar ── */
.score-bar-outer {
    background: var(--surface2);
    border-radius: 2px;
    height: 6px;
    margin-top: 4px;
}
.score-bar-inner {
    height: 6px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--blue));
}

/* ── Header ── */
.terminal-header {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--muted);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.ticker-display {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.price-display {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    color: var(--green);
}

/* ── Table overrides ── */
.stDataFrame { background: var(--surface) !important; }

/* ── Remove streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Status/Progress ── */
.stStatus { background: var(--surface) !important; border: 1px solid var(--border) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: var(--surface); gap: 0; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    padding: 0.6rem 1.2rem;
}
.stTabs [aria-selected="true"] { color: var(--text) !important; border-bottom: 2px solid var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Imports (after path setup) ───────────────────────────────────────────────
from graph.graph import finsight_graph
from vector_store.store import memory_store
from graph.state import FinancialState


# ─── Helper Functions ─────────────────────────────────────────────────────────
def color_value(val, positive_good=True):
    """Return green/red/yellow class based on numeric value."""
    if val is None:
        return "metric-value"
    try:
        f = float(str(val).replace('%', '').replace('$', ''))
        if positive_good:
            return "metric-value metric-green" if f > 0 else "metric-value metric-red"
        else:
            return "metric-value metric-red" if f > 0 else "metric-value metric-green"
    except:
        return "metric-value"

def fmt(val, prefix="", suffix="", decimals=2, fallback="N/A"):
    if val is None:
        return fallback
    try:
        return f"{prefix}{float(val):.{decimals}f}{suffix}"
    except:
        return str(val)

def pct(val):
    if val is None: return "N/A"
    try: return f"{float(val)*100:.1f}%"
    except: return str(val)

def score_bar(score: float, label: str):
    pct_width = min(100, max(0, score * 10))
    color = "#00d68f" if score >= 7 else "#ffb400" if score >= 5 else "#ff3b5c"
    return f"""
    <div style="margin-bottom:0.6rem">
        <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#5a6d8a;margin-bottom:3px">
            <span>{label}</span><span style="color:{color}">{score:.1f}/10</span>
        </div>
        <div class="score-bar-outer">
            <div class="score-bar-inner" style="width:{pct_width}%;background:{color}"></div>
        </div>
    </div>"""

def rec_color(rec: str) -> str:
    r = rec.upper()
    if "STRONG BUY" in r or r == "BUY": return "#00d68f"
    if "STRONG SELL" in r or r == "SELL": return "#ff3b5c"
    return "#ffb400"

def winner_emoji(winner: str) -> str:
    return {"bull": "🐂", "bear": "🐻", "neutral": "⚖️"}.get(winner, "⚖️")


# ─── Build price chart ────────────────────────────────────────────────────────
def build_price_chart(price_data: dict, technicals: dict) -> go.Figure:
    records = price_data.get("history_records", [])
    if not records:
        return go.Figure()

    df = pd.DataFrame(records)
    df.columns = [c.replace(' 00:00:00+00:00', '') for c in df.columns]

    date_col  = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    close_col = next((c for c in df.columns if "close" in c.lower()), "Close")

    fig = go.Figure()

    # Candlestick
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        fig.add_trace(go.Candlestick(
            x=df[date_col], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#00d68f", decreasing_line_color="#ff3b5c",
            name="Price",
            increasing_fillcolor="rgba(0,214,143,0.3)",
            decreasing_fillcolor="rgba(255,59,92,0.3)",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[close_col],
            line=dict(color="#4d9fff", width=1.5), name="Price",
        ))

    # SMAs
    sma50  = technicals.get("sma50")
    sma200 = technicals.get("sma200")
    if sma50:
        sma50_series = df[close_col].rolling(50).mean()
        fig.add_trace(go.Scatter(x=df[date_col], y=sma50_series,
            line=dict(color="#ffb400", width=1, dash="dot"), name="SMA50"))
    if sma200:
        sma200_series = df[close_col].rolling(200).mean()
        fig.add_trace(go.Scatter(x=df[date_col], y=sma200_series,
            line=dict(color="#7c5cfc", width=1, dash="dot"), name="SMA200"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            showgrid=False, color="#5a6d8a",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(showgrid=True, gridcolor="#1e2d50", color="#5a6d8a"),
        legend=dict(
            bgcolor="rgba(15,22,41,0.8)", font=dict(color="#c8d8f0", size=10),
            x=0.01, y=0.99,
        ),
        hovermode="x unified",
        font=dict(family="IBM Plex Mono", color="#5a6d8a"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0">
        <div class="terminal-header">FinSight Alpha v1.0</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:700;color:#c8d8f0">
            Multi-Agent<br/>Financial AI
        </div>
        <div style="color:#5a6d8a;font-size:0.8rem;margin-top:0.5rem;line-height:1.5">
            Adversarial debate architecture.<br/>
            Bull vs Bear. Judge arbitrates.<br/>
            Vector memory. Self-evaluation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    ticker_input = st.text_input(
        "TICKER SYMBOL",
        value="AAPL",
        placeholder="AAPL, TSLA, NVDA...",
    ).upper().strip()

    question_input = st.text_area(
        "ANALYSIS QUESTION",
        value="Should I buy this stock for a 6-month hold? What are the key risks?",
        height=100,
    )

    api_key = st.text_input(
        "GROQ API KEY",
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com — no card needed",
        value=os.getenv("GROQ_API_KEY", ""),
    )

    run_btn = st.button("▶ RUN ANALYSIS", type="primary")

    st.divider()

    # Memory Stats
    try:
        count = memory_store.count()
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Vector Memory</div>
            <div style="font-family:'IBM Plex Mono',monospace;color:#4d9fff;font-size:1.2rem">{count}</div>
            <div style="color:#5a6d8a;font-size:0.75rem">analyses stored</div>
        </div>
        """, unsafe_allow_html=True)
    except:
        pass

    st.markdown("""
    <div class="card">
        <div class="card-header">Graph Architecture</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#5a6d8a;line-height:1.8">
            START<br/>
            &nbsp;&nbsp;↓ data_node<br/>
            &nbsp;&nbsp;↓ memory_node<br/>
            &nbsp;&nbsp;↓ bull_node 🐂<br/>
            &nbsp;&nbsp;↓ bear_node 🐻<br/>
            &nbsp;&nbsp;↓ judge_node ⚖️<br/>
            &nbsp;&nbsp;↓ synthesizer<br/>
            &nbsp;&nbsp;↓ evaluator<br/>
            &nbsp;&nbsp;↓ store_node<br/>
            END
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid #1e2d50">
    <div>
        <div class="terminal-header">FINSIGHT ALPHA // MULTI-AGENT INTELLIGENCE SYSTEM</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:700;color:#c8d8f0;letter-spacing:-0.02em">
            Adversarial Debate Architecture
        </div>
        <div style="color:#5a6d8a;font-size:0.85rem;margin-top:0.3rem">
            LangGraph · ChromaDB · Claude claude-sonnet-4 · yfinance · Technical Analysis
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Analysis Logic ───────────────────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar. Get one free at console.groq.com")
    elif not ticker_input:
        st.error("⚠️ Please enter a ticker symbol.")
    else:
        # Set API key for this session
        os.environ["GROQ_API_KEY"] = api_key

        import config as conf_mod
        conf_mod.GROQ_API_KEY = api_key

        # Reinitialize the Groq client with the new key
        import agents.nodes as nodes_mod
        from groq import Groq
        nodes_mod.groq_client = Groq(api_key=api_key)

        initial_state: FinancialState = {
            "ticker":          ticker_input,
            "question":        question_input,
            "price_data":      {},
            "technicals":      {},
            "fundamentals":    {},
            "news_headlines":  [],
            "past_analyses":   [],
            "memory_context":  "",
            "bull_case":       "",
            "bull_confidence": 0.0,
            "bear_case":       "",
            "bear_confidence": 0.0,
            "verdict":         "",
            "winner":          "neutral",
            "judge_confidence": 0.0,
            "risk_metrics":    {},
            "risk_level":      "MEDIUM",
            "risk_commentary": "",
            "recommendation":  "HOLD",
            "price_target":    None,
            "final_summary":   "",
            "reasoning_score": 0.0,
            "evidence_score":  0.0,
            "overall_score":   0.0,
            "eval_notes":      "",
            "analysis_id":     "",
            "errors":          [],
            "steps_completed": [],
        }

        # ── Execution Trace UI ────────────────────────────────────────────────
        exec_col, _ = st.columns([1, 3])
        with exec_col:
            st.markdown('<div class="card"><div class="card-header">⚡ Agent Execution Trace</div>', unsafe_allow_html=True)
            trace_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        trace_steps = []
        final_state = None

        NODE_LABELS = {
            "data":        "📊 Fetching Market Data",
            "memory":      "🧠 Querying Vector Memory",
            "bull":        "🐂 Bull Agent Analyzing",
            "bear":        "🐻 Bear Agent Analyzing",
            "judge":       "⚖️  Judge Arbitrating Debate",
            "synthesizer": "🎯 Synthesizing Recommendation",
            "evaluator":   "📋 Evaluating Analysis Quality",
            "store":       "💾 Persisting to Memory",
        }

        # Stream the graph execution
        for event in finsight_graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                label = NODE_LABELS.get(node_name, f"▶ {node_name}")
                steps = node_output.get("steps_completed", [])
                if steps:
                    trace_steps.append(steps[-1])
                else:
                    trace_steps.append(f"✓ {label}")

                # Update trace display
                trace_html = "".join([
                    f'<div class="step-item" style="animation-delay:{i*0.1}s">{s}</div>'
                    for i, s in enumerate(trace_steps)
                ])
                trace_placeholder.markdown(
                    f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.78rem">{trace_html}</div>',
                    unsafe_allow_html=True
                )

                # Accumulate state
                if final_state is None:
                    final_state = {**initial_state, **node_output}
                else:
                    final_state = {**final_state, **node_output}

        if final_state is None:
            st.error("Analysis failed to run.")
            st.stop()

        st.success(f"✅ Analysis complete — {len(trace_steps)} agents ran successfully")

        # ═══════════════════════════════════════════════════════════════════════
        # RENDER RESULTS
        # ═══════════════════════════════════════════════════════════════════════
        pd_data  = final_state.get("price_data", {})
        tech     = final_state.get("technicals", {})
        fund     = final_state.get("fundamentals", {})
        risk     = final_state.get("risk_metrics", {})
        news     = final_state.get("news_headlines", [])
        rec      = final_state.get("recommendation", "HOLD")
        target   = final_state.get("price_target")
        winner   = final_state.get("winner", "neutral")

        # ── Header: Ticker + Price + Recommendation ──────────────────────────
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            price = pd_data.get("current_price", "N/A")
            ytd   = pd_data.get("ytd_return_pct", 0) or 0
            ytd_color = "#00d68f" if ytd >= 0 else "#ff3b5c"
            ytd_sign  = "+" if ytd >= 0 else ""
            st.markdown(f"""
            <div class="card">
                <div class="terminal-header">{pd_data.get('company_name', ticker_input)}</div>
                <div class="ticker-display">{ticker_input}</div>
                <div class="price-display">${price}</div>
                <div style="font-family:'IBM Plex Mono',monospace;color:{ytd_color};font-size:0.9rem;margin-top:0.3rem">
                    {ytd_sign}{ytd:.2f}% YTD &nbsp;|&nbsp; {pd_data.get('sector','N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            rc = rec_color(rec)
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:1.5rem">
                <div class="card-header">RECOMMENDATION</div>
                <div style="display:inline-block;padding:0.5rem 1rem;background:rgba(0,0,0,0.3);
                     border:2px solid {rc};border-radius:4px;
                     font-family:'IBM Plex Mono',monospace;font-size:1.3rem;
                     font-weight:700;color:{rc};letter-spacing:0.05em">
                    {rec}
                </div>
                {'<div style="margin-top:0.5rem;font-family:IBM Plex Mono,monospace;color:#4d9fff;font-size:0.9rem">Target: $' + f"{target:.2f}" + '</div>' if target else ''}
            </div>
            """, unsafe_allow_html=True)

        with col3:
            w_color = "#00d68f" if winner == "bull" else "#ff3b5c" if winner == "bear" else "#ffb400"
            o_score = final_state.get("overall_score", 0)
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:1.5rem">
                <div class="card-header">DEBATE WINNER</div>
                <div style="font-size:2rem">{winner_emoji(winner)}</div>
                <div style="font-family:'IBM Plex Mono',monospace;color:{w_color};font-weight:700;font-size:1rem">
                    {winner.upper()}
                </div>
                <div style="margin-top:0.5rem;color:#5a6d8a;font-size:0.75rem">
                    Quality Score: <span style="color:#7c5cfc">{o_score:.1f}/10</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Tabs ──────────────────────────────────────────────────────────────
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 CHART", "🥊 THE DEBATE", "⚖️ VERDICT", "🎯 SYNTHESIS", "📊 METRICS", "🧠 MEMORY"
        ])

        # ── TAB 1: Chart ──────────────────────────────────────────────────────
        with tab1:
            fig = build_price_chart(pd_data, tech)
            st.plotly_chart(fig, use_container_width=True)

            # Key metrics grid
            metrics = [
                ("52W HIGH",    f"${pd_data.get('52w_high','N/A')}",    False),
                ("52W LOW",     f"${pd_data.get('52w_low','N/A')}",     False),
                ("% FROM HIGH", f"{pd_data.get('pct_from_high','N/A')}%", True),
                ("RSI",         f"{tech.get('rsi','N/A')}",              False),
                ("MACD",        f"{'▲' if tech.get('macd_bullish') else '▼'} {tech.get('macd_histogram','N/A')}", False),
                ("SMA50",       f"${tech.get('sma50','N/A')}",           False),
                ("SMA200",      f"${tech.get('sma200','N/A')}",          False),
                ("VOL TREND",   tech.get("volume_trend","N/A").upper(),  False),
            ]
            cols = st.columns(8)
            for i, (label, val, inv) in enumerate(metrics):
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{val}</div>
                    </div>""", unsafe_allow_html=True)

        # ── TAB 2: The Debate ─────────────────────────────────────────────────
        with tab2:
            st.markdown("""
            <div style="color:#5a6d8a;font-size:0.8rem;font-family:'IBM Plex Mono',monospace;margin-bottom:1rem">
            ◆ TWO INDEPENDENT AI ANALYSTS — ONE BULL, ONE BEAR — GIVEN THE SAME DATA.
            NEITHER SEES THE OTHER'S CASE UNTIL THE JUDGE REVIEWS BOTH.
            </div>""", unsafe_allow_html=True)

            bc  = final_state.get("bull_confidence", 0)
            brc = final_state.get("bear_confidence", 0)
            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="bull-card">
                    <div class="bull-title">🐂 BULL CASE — CONFIDENCE: {bc:.1f}/10</div>
                    <div style="height:6px;background:rgba(0,214,143,0.15);border-radius:3px;margin:0.5rem 0">
                        <div style="width:{bc*10}%;height:100%;background:#00d68f;border-radius:3px"></div>
                    </div>
                    <div style="color:#c8d8f0;font-size:0.85rem;line-height:1.7;margin-top:0.8rem;white-space:pre-wrap">
                        {final_state.get('bull_case', 'No bull case generated.')}
                    </div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="bear-card">
                    <div class="bear-title">🐻 BEAR CASE — CONFIDENCE: {brc:.1f}/10</div>
                    <div style="height:6px;background:rgba(255,59,92,0.15);border-radius:3px;margin:0.5rem 0">
                        <div style="width:{brc*10}%;height:100%;background:#ff3b5c;border-radius:3px"></div>
                    </div>
                    <div style="color:#c8d8f0;font-size:0.85rem;line-height:1.7;margin-top:0.8rem;white-space:pre-wrap">
                        {final_state.get('bear_case', 'No bear case generated.')}
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── TAB 3: Judge's Verdict ────────────────────────────────────────────
        with tab3:
            w_color = "#00d68f" if winner == "bull" else "#ff3b5c" if winner == "bear" else "#ffb400"
            jc = final_state.get("judge_confidence", 0)
            st.markdown(f"""
            <div class="card">
                <div class="card-header">⚖️ JUDGE'S RULING — {winner.upper()} CASE PREVAILS</div>
                <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem">
                    <div style="font-size:3rem">{winner_emoji(winner)}</div>
                    <div>
                        <div style="font-family:'IBM Plex Mono',monospace;color:{w_color};font-size:1.2rem;font-weight:700">
                            {winner.upper()} CASE WINS
                        </div>
                        <div style="color:#5a6d8a;font-size:0.8rem">Judge confidence: {jc:.1f}/10</div>
                    </div>
                </div>
                <div style="color:#c8d8f0;font-size:0.85rem;line-height:1.7;white-space:pre-wrap">
                    {final_state.get('verdict', 'No verdict generated.')}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── TAB 4: Final Synthesis ────────────────────────────────────────────
        with tab4:
            rc = rec_color(rec)
            st.markdown(f"""
            <div class="card">
                <div class="card-header">🎯 FINAL RECOMMENDATION</div>
                <div style="display:flex;align-items:center;gap:1.5rem;margin-bottom:1.2rem;flex-wrap:wrap">
                    <div style="padding:0.5rem 1.5rem;border:2px solid {rc};border-radius:4px;
                         font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:700;color:{rc}">
                        {rec}
                    </div>
                    {'<div style="font-family:IBM Plex Mono,monospace;font-size:1.2rem;color:#4d9fff">12M Target: $' + f"{target:.2f}" + '</div>' if target else ''}
                </div>
                <div style="color:#c8d8f0;font-size:0.85rem;line-height:1.7;white-space:pre-wrap">
                    {final_state.get('final_summary', 'No synthesis generated.')}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── TAB 5: Metrics ────────────────────────────────────────────────────
        with tab5:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown('<div class="card"><div class="card-header">📐 Fundamentals</div>', unsafe_allow_html=True)
                fund_metrics = [
                    ("P/E Ratio",      fmt(fund.get("pe_ratio"), decimals=1)),
                    ("Forward P/E",    fmt(fund.get("forward_pe"), decimals=1)),
                    ("PEG Ratio",      fmt(fund.get("peg_ratio"), decimals=2)),
                    ("Price/Book",     fmt(fund.get("price_to_book"), decimals=2)),
                    ("EV/EBITDA",      fmt(fund.get("ev_to_ebitda"), decimals=1)),
                    ("Revenue Growth", pct(fund.get("revenue_growth"))),
                    ("Profit Margin",  pct(fund.get("profit_margin"))),
                    ("ROE",            pct(fund.get("roe"))),
                    ("Debt/Equity",    fmt(fund.get("debt_to_equity"), decimals=2)),
                ]
                for label, val in fund_metrics:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;padding:0.3rem 0;
                         border-bottom:1px solid #1e2d50;font-size:0.8rem;font-family:'IBM Plex Mono',monospace">
                        <span style="color:#5a6d8a">{label}</span>
                        <span style="color:#c8d8f0">{val}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                risk_color = {"LOW": "#00d68f", "MEDIUM": "#ffb400", "HIGH": "#ff3b5c"}.get(
                    risk.get("risk_level", "MEDIUM"), "#ffb400")
                st.markdown(f'<div class="card"><div class="card-header">⚠️ Risk Metrics</div>', unsafe_allow_html=True)
                risk_metrics = [
                    ("Risk Level",   risk.get("risk_level", "N/A")),
                    ("Annual Volatility", fmt(risk.get("annualized_volatility"), suffix="%", decimals=1)),
                    ("Max Drawdown",      fmt(risk.get("max_drawdown"), suffix="%", decimals=1)),
                    ("Sharpe Ratio",      fmt(risk.get("sharpe_ratio"), decimals=2)),
                    ("Beta",              fmt(risk.get("beta"), decimals=2)),
                    ("YTD Return",        fmt(pd_data.get("ytd_return_pct"), suffix="%", decimals=2)),
                    ("Short % Float",     pct(fund.get("short_percent_float"))),
                ]
                for label, val in risk_metrics:
                    is_risk = label == "Risk Level"
                    color = risk_color if is_risk else "#c8d8f0"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;padding:0.3rem 0;
                         border-bottom:1px solid #1e2d50;font-size:0.8rem;font-family:'IBM Plex Mono',monospace">
                        <span style="color:#5a6d8a">{label}</span>
                        <span style="color:{color};font-weight:{'700' if is_risk else '400'}">{val}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c3:
                r_score = final_state.get("reasoning_score", 0)
                e_score = final_state.get("evidence_score", 0)
                o_score = final_state.get("overall_score", 0)
                st.markdown(f"""
                <div class="card">
                    <div class="card-header">📋 Analysis Quality</div>
                    {score_bar(r_score, "Reasoning Logic")}
                    {score_bar(e_score, "Evidence Quality")}
                    {score_bar(o_score, "Overall Score")}
                    <div style="margin-top:1rem;font-size:0.8rem;color:#5a6d8a;line-height:1.6">
                        {final_state.get('eval_notes', 'No evaluation notes.')[:300]}
                    </div>
                </div>""", unsafe_allow_html=True)

            # News
            if news:
                st.markdown('<div class="card"><div class="card-header">📰 Recent News</div>', unsafe_allow_html=True)
                for n in news[:6]:
                    title = n.get("title", "")
                    pub   = n.get("publisher", "")
                    link  = n.get("link", "#")
                    st.markdown(f"""
                    <div style="padding:0.5rem 0;border-bottom:1px solid #1e2d50">
                        <a href="{link}" target="_blank"
                           style="color:#4d9fff;font-size:0.82rem;text-decoration:none;font-family:'IBM Plex Sans',sans-serif">
                            {title}
                        </a>
                        <div style="color:#5a6d8a;font-size:0.7rem;font-family:'IBM Plex Mono',monospace;margin-top:2px">{pub}</div>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── TAB 6: Memory Browser ─────────────────────────────────────────────
        with tab6:
            st.markdown("""
            <div style="color:#5a6d8a;font-size:0.8rem;font-family:'IBM Plex Mono',monospace;margin-bottom:1rem">
            ◆ CHROMADB VECTOR STORE — ALL PAST ANALYSES EMBEDDED AND INDEXED.
            AGENTS RETRIEVE RELEVANT CONTEXT ON EACH NEW ANALYSIS.
            </div>""", unsafe_allow_html=True)

            past = final_state.get("past_analyses", [])
            all_analyses = memory_store.get_all_analyses(limit=20)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="card"><div class="card-header">Context Used in This Analysis ({len(past)} retrieved)</div>', unsafe_allow_html=True)
                if past:
                    for p in past:
                        r = p.get("recommendation", "HOLD")
                        rc2 = rec_color(r)
                        st.markdown(f"""
                        <div style="padding:0.6rem 0;border-bottom:1px solid #1e2d50">
                            <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:0.75rem">
                                <span style="color:#c8d8f0">{p.get('ticker')} — {p.get('timestamp','')}</span>
                                <span style="color:{rc2};font-weight:700">{r}</span>
                            </div>
                            <div style="color:#5a6d8a;font-size:0.75rem;margin-top:3px">
                                Risk: {p.get('risk_level','N/A')} · Score: {p.get('quality_score','N/A')}/10
                            </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#5a6d8a;font-size:0.8rem">No past analyses in memory yet.</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown(f'<div class="card"><div class="card-header">All Stored Analyses ({len(all_analyses)} total)</div>', unsafe_allow_html=True)
                if all_analyses:
                    for p in all_analyses[:10]:
                        r = p.get("recommendation", "HOLD")
                        rc2 = rec_color(r)
                        st.markdown(f"""
                        <div style="padding:0.5rem 0;border-bottom:1px solid #1e2d50">
                            <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:0.75rem">
                                <span style="color:#c8d8f0">{p.get('ticker')} — {p.get('timestamp','')[:10]}</span>
                                <span style="color:{rc2}">{r}</span>
                            </div>
                            <div style="color:#5a6d8a;font-size:0.7rem;margin-top:2px">{p.get('document','')[:120]}...</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#5a6d8a;font-size:0.8rem">Memory is empty. Run your first analysis!</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

else:
    # ── Landing state ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;opacity:0.6">
        <div style="font-size:4rem;margin-bottom:1rem">📈</div>
        <div style="font-family:'IBM Plex Mono',monospace;color:#5a6d8a;font-size:1rem;letter-spacing:0.1em">
            ENTER A TICKER → RUN ANALYSIS<br/>
            <span style="font-size:0.8rem;opacity:0.6">
                Bull & Bear agents debate · Judge arbitrates · Vector memory persists
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show architecture explainer
    col1, col2, col3 = st.columns(3)
    for col, emoji, title, desc in [
        (col1, "🥊", "Adversarial Debate", "Bull and Bear agents independently analyze the same data, then argue opposing cases. A neutral Judge evaluates which argument is more rigorous."),
        (col2, "🧠", "Vector Memory", "ChromaDB stores every analysis as an embedding. Future analyses retrieve relevant past context — the system gets smarter over time."),
        (col3, "📋", "Self-Evaluation", "A meta-evaluator scores reasoning quality, evidence use, and logical coherence. Low-quality analyses are flagged — trust is quantified."),
    ]:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:1.5rem">
                <div style="font-size:2rem;margin-bottom:0.5rem">{emoji}</div>
                <div style="font-family:'IBM Plex Mono',monospace;color:#c8d8f0;font-size:0.85rem;font-weight:600;margin-bottom:0.5rem">{title}</div>
                <div style="color:#5a6d8a;font-size:0.8rem;line-height:1.5">{desc}</div>
            </div>""", unsafe_allow_html=True)