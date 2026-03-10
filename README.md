# FinSight Alpha 📈
### Multi-Agent Financial Intelligence System

> **Adversarial Debate Architecture** — Two AI analysts argue Bull vs Bear, a Judge arbitrates, 
> vector memory persists every analysis, and a meta-evaluator scores reasoning quality.

---

## 🏗️ Architecture

```
INPUT (ticker + question)
         │
         ▼
┌─────────────────┐
│  data_node      │  ← yfinance: OHLCV, fundamentals, news, technicals
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  memory_node    │  ← ChromaDB: retrieve past analyses as context
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  bull_node  🐂  │      │  bear_node  🐻  │  ← Sequential (parallelizable)
│  "Buy thesis"   │      │  "Short thesis" │
└────────┬────────┘      └────────┬────────┘
         │                        │
         └──────────┬─────────────┘
                    │
                    ▼
         ┌─────────────────┐
         │  judge_node  ⚖️  │  ← Reads BOTH cases, rules who wins
         └────────┬────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ synthesizer_node │  ← Final BUY/SELL/HOLD + price target
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  evaluator_node  │  ← Scores reasoning quality (0-10)
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │   store_node     │  ← Embeds + persists to ChromaDB
         └──────────────────┘
```

## 🧠 What Makes This Unique

| Feature | Most Demos | FinSight Alpha |
|---------|-----------|----------------|
| Analyst perspective | Single agent | Bull vs Bear debate |
| Memory | None | ChromaDB vector store |
| Quality assurance | None | Self-evaluation scoring |
| Reasoning trace | Hidden | Full agent trace in UI |
| Data | Mock/static | Live yfinance data |
| Technical analysis | None | RSI, MACD, BB, SMA |

## 📚 Concepts You'll Learn

1. **LangGraph StateGraph** — Define nodes + edges, state flows through the graph
2. **Agent Design** — Each agent has a persona, role, and specific system prompt
3. **Vector Databases** — ChromaDB stores embeddings for semantic retrieval
4. **Adversarial AI** — Debate architecture forces rigorous argument
5. **Meta-evaluation** — AI scoring AI output (key production pattern)
6. **Graceful Degradation** — Errors are caught, stored in state, don't crash the system
7. **Streaming** — LangGraph `.stream()` enables real-time UI updates

## 🚀 Setup

```bash
# 1. Clone and navigate
cd finsight

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env

# 4. Run
streamlit run app.py
```

## 🔑 Required

- **Anthropic API key** — Get at [console.anthropic.com](https://console.anthropic.com)
- Python 3.10+
- Internet connection (for yfinance)

## 📁 Project Structure

```
finsight/
├── app.py                  # Streamlit UI (Bloomberg Terminal aesthetic)
├── config.py               # API keys, model settings
├── requirements.txt
├── agents/
│   └── nodes.py            # All 8 agent node functions
├── graph/
│   ├── state.py            # LangGraph TypedDict state schema
│   └── graph.py            # StateGraph definition + compilation
├── tools/
│   └── financial_tools.py  # yfinance + technical analysis
└── vector_store/
    └── store.py            # ChromaDB operations
```

## 🔬 Extending The System

```python
# Add a new node in agents/nodes.py
def macro_node(state: FinancialState) -> Dict:
    """Add macroeconomic context from web search."""
    ...

# Register in graph/graph.py
graph.add_node("macro", macro_node)
graph.add_edge("memory", "macro")
graph.add_edge("macro", "bull")  # macro runs before debate
```

## 📊 UI Tabs

| Tab | Contents |
|-----|----------|
| 📈 CHART | Interactive candlestick + SMA overlays |
| 🥊 THE DEBATE | Bull vs Bear cases side by side |
| ⚖️ VERDICT | Judge's ruling with confidence score |
| 🎯 SYNTHESIS | Final recommendation + price target |
| 📊 METRICS | Fundamentals, risk metrics, news |
| 🧠 MEMORY | ChromaDB browser, past analyses |

---

*Built with LangGraph · Anthropic Claude · ChromaDB · yfinance · Streamlit*