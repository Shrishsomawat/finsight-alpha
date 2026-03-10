"""
graph/graph.py
───────────────
EDUCATIONAL NOTE: This is the HEART of LangGraph — the StateGraph.

A StateGraph defines:
  - NODES: Python functions that process state
  - EDGES: How control flows between nodes

The execution model is simple but powerful:
  1. Each node runs in sequence
  2. Each node returns a dict of state updates
  3. LangGraph merges updates into the shared state
  4. The graph runs until it hits END

Our graph looks like:
  START
    ↓
  data_node       ← Pure data fetching (yfinance)
    ↓
  memory_node     ← Vector DB retrieval (ChromaDB)
    ↓
  bull_node       ← LLM: Bullish thesis
    ↓
  bear_node       ← LLM: Bearish thesis
    ↓
  judge_node      ← LLM: Debate arbitration
    ↓
  synthesizer_node ← LLM: Final recommendation
    ↓
  evaluator_node  ← LLM: Quality scoring
    ↓
  store_node      ← Persist to ChromaDB
    ↓
  END

Future extension: bull_node and bear_node could run in PARALLEL
using LangGraph's Send API for fan-out/fan-in patterns.
"""
from langgraph.graph import StateGraph, START, END
from graph.state import FinancialState
from agents.nodes import (
    data_node,
    memory_node,
    bull_node,
    bear_node,
    judge_node,
    synthesizer_node,
    evaluator_node,
    store_node,
)


def build_graph():
    """
    Build and compile the FinSight financial analysis graph.

    Returns a compiled LangGraph that can be:
      - .invoke(state)  → run synchronously, get final state
      - .stream(state)  → stream state updates node-by-node (used in Streamlit)
    """
    # Initialize the graph with our state schema
    # The state schema tells LangGraph what fields to expect/track
    graph = StateGraph(FinancialState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    # Each node is registered with a name and a function
    graph.add_node("data",        data_node)
    graph.add_node("memory",      memory_node)
    graph.add_node("bull",        bull_node)
    graph.add_node("bear",        bear_node)
    graph.add_node("judge",       judge_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("evaluator",   evaluator_node)
    graph.add_node("store",       store_node)

    # ── Define edges (control flow) ────────────────────────────────────────────
    # Simple sequential flow for clarity
    # In production you might add conditional edges:
    #   graph.add_conditional_edges("judge", route_by_winner, {...})
    graph.add_edge(START,       "data")
    graph.add_edge("data",      "memory")
    graph.add_edge("memory",    "bull")
    graph.add_edge("bull",      "bear")
    graph.add_edge("bear",      "judge")
    graph.add_edge("judge",     "synthesizer")
    graph.add_edge("synthesizer", "evaluator")
    graph.add_edge("evaluator", "store")
    graph.add_edge("store",     END)

    # Compile — this validates the graph and creates the runnable
    return graph.compile()


# Build the graph once at module import time
# (LangGraph compilation is expensive, do it once)
finsight_graph = build_graph()