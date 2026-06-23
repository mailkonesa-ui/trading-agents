"""
Streamlit UI for the trading-agents system.

Layout:
  - Sidebar: universe picker, agent parameter sliders, learning settings
  - Main:    scan all tickers, show top BUY/SELL proposals, accept/reject
  - Tabs:    Charts, History, Agent performance, Settings
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make `core` importable regardless of how Streamlit is launched
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import agents as agents_mod
from core import data, learning, storage, universe


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
storage.init_db()
st.set_page_config(page_title="Trading Agents", page_icon="📈", layout="wide")
st.title("📈 Trading Agents — Learning Portfolio")
st.caption(
    "Multiple strategy agents propose BUY/SELL. You decide. "
    "Agents learn from realised P&L on the proposals you accept."
)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("ticker", "AAPL")
    st.session_state.setdefault("agent_params", dict(agents_mod.DEFAULT_PARAMS))
    # load saved params if any
    for name in agents_mod.AGENTS:
        saved = storage.load_params(name)
        if saved:
            st.session_state.agent_params[name] = saved


_init_state()


# ---------------------------------------------------------------------------
# Sidebar — universe + agent tuning
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🎯 Ticker")
    group = st.selectbox("Group", list(universe.UNIVERSE.keys()))
    sym, name = st.selectbox(
        "Symbol",
        universe.UNIVERSE[group],
        format_func=lambda x: f"{x[0]} — {x[1]}",
    )
    st.session_state.ticker = sym

    st.divider()
    st.header("🧠 Agents (tunable)")

    enabled: dict[str, bool] = {}
    for agent_name, fn in agents_mod.AGENTS.items():
        with st.expander(agent_name, expanded=False):
            enabled[agent_name] = st.checkbox(
                f"Enable {agent_name}",
                value=True,
                key=f"en_{agent_name}",
            )
            bounds = agents_mod.PARAM_BOUNDS[agent_name]
            current = st.session_state.agent_params[agent_name]
            new_params: dict = {}
            for pname, (lo, hi) in bounds.items():
                if isinstance(lo, float) or isinstance(hi, float):
                    step = (hi - lo) / 100
                    new_params[pname] = st.slider(
                        pname, lo, hi, float(current.get(pname, lo)), step=step,
                        key=f"{agent_name}_{pname}",
                    )
                else:
                    new_params[pname] = st.slider(
                        pname, int(lo), int(hi), int(current.get(pname, lo)),
                        key=f"{agent_name}_{pname}",
                    )
            st.session_state.agent_params[agent_name] = new_params
            storage.save_params(agent_name, new_params)

    st.divider()
    st.header("📚 Learning")
    alpha = st.slider(
        "Learning rate (alpha)",
        min_value=0.05, max_value=2.0, value=0.5, step=0.05,
        help="How aggressively agent weights shift after a closed trade.",
    )
    min_confidence = st.slider(
        "Minimum confidence to surface a proposal",
        min_value=0.0, max_value=1.0, value=0.55, step=0.05,
    )
    st.caption(
        "Agents do NOT train on price data. They only learn from the "
        "P&L of proposals you accept."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def _history(ticker: str) -> pd.DataFrame:
    return data.combined_series(ticker)


def _scan_one(ticker: str, params_by_agent: dict, enabled_agents: dict) -> list[dict]:
    df = _history(ticker)
    if df.empty or len(df) < 35:
        return []
    proposals: list[dict] = []
    for agent_name in agents_mod.AGENTS:
        if not enabled_agents.get(agent_name, True):
            continue
        params = params_by_agent.get(agent_name, agents_mod.DEFAULT_PARAMS[agent_name])
        action, conf, reason = agents_mod.run_agent(agent_name, df, params)
        if action in ("BUY", "SELL") and conf >= min_confidence:
            stats = storage.get_agent_stats(agent_name)
            proposals.append(
                {
                    "ticker": ticker,
                    "agent": agent_name,
                    "action": action,
                    "confidence": float(conf),
                    "reason": reason,
                    "price": float(df["Close"].iloc[-1]),
                    "weight": float(stats["weight"]),
                    "wins": int(stats["wins"]),
                    "losses": int(stats["losses"]),
                }
            )
    return proposals


def _score(p: dict) -> float:
    """Rank proposals by confidence * agent weight."""
    return p["confidence"] * p["weight"]


# ---------------------------------------------------------------------------
# Main: scan all + current ticker detail
# ---------------------------------------------------------------------------
col_scan, col_detail = st.columns([1.2, 2])

with col_scan:
    st.subheader("🔎 Scan universe")
    if st.button("Rescan all tickers", type="primary", use_container_width=True):
        with st.spinner("Scanning 24 tickers across enabled agents..."):
            all_props: list[dict] = []
            for t in universe.all_tickers():
                all_props.extend(
                    _scan_one(t, st.session_state.agent_params, enabled)
                )
            all_props.sort(key=_score, reverse=True)
            st.session_state["scan_results"] = all_props
    else:
        st.session_state.setdefault("scan_results", [])

    results = st.session_state.get("scan_results", [])
    st.caption(f"{len(results)} active proposals")
    for p in results[:12]:
        s = _score(p)
        c_action = "🟢" if p["action"] == "BUY" else "🔴"
        with st.container(border=True):
            st.markdown(
                f"**{c_action} {p['action']} {p['ticker']}** "
                f"<small>({p['agent']})</small>",
                unsafe_allow_html=True,
            )
            st.write(f"Score: `{s:.2f}` • Conf: `{p['confidence']:.2f}` • "
                     f"Agent weight: `{p['weight']:.2f}`")
            st.caption(f"€{p['price']:.2f} — {p['reason']}")
            cc1, cc2 = st.columns(2)
            if cc1.button("Accept", key=f"acc_{p['ticker']}_{p['agent']}_{p['action']}",
                          use_container_width=True):
                # close any open trade for this ticker first
                learning.close_previous_if_open(p["ticker"], p["price"])
                pid = storage.insert_proposal(
                    p["ticker"], p["agent"], p["action"],
                    p["confidence"], p["reason"], p["price"],
                )
                storage.decide_proposal(pid, "ACCEPT")
                st.toast(f"Accepted {p['action']} {p['ticker']} ({p['agent']})")
                st.rerun()
            if cc2.button("Reject", key=f"rej_{p['ticker']}_{p['agent']}_{p['action']}",
                          use_container_width=True):
                pid = storage.insert_proposal(
                    p["ticker"], p["agent"], p["action"],
                    p["confidence"], p["reason"], p["price"],
                )
                storage.decide_proposal(pid, "REJECT")
                st.toast(f"Rejected {p['action']} {p['ticker']} ({p['agent']})")
                st.rerun()

with col_detail:
    st.subheader(f"{name} ({sym})")
    try:
        df = _history(sym)
    except Exception as e:
        st.error(f"Could not load data for {sym}: {e}")
        st.stop()

    if df.empty:
        st.warning("No data available.")
        st.stop()

    # Price chart with 20d SMA
    sma20 = df["Close"].rolling(20).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
    ))
    fig.add_trace(go.Scatter(x=df.index, y=sma20, mode="lines",
                             name="SMA20", line=dict(width=1)))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Run agents on this ticker
    st.markdown("### Agent signals on this ticker")
    rows = []
    for agent_name in agents_mod.AGENTS:
        params = st.session_state.agent_params[agent_name]
        action, conf, reason = agents_mod.run_agent(agent_name, df, params)
        stats = storage.get_agent_stats(agent_name)
        rows.append({
            "Agent": agent_name,
            "Signal": action,
            "Confidence": round(conf, 2),
            "Weight": round(stats["weight"], 2),
            "W/L": f"{stats['wins']}/{stats['losses']}",
            "Reason": reason,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Force a manual proposal for the current ticker if you want
    st.markdown("### Manual proposal (current ticker)")
    mc1, mc2, mc3 = st.columns(3)
    if mc1.button("Propose BUY", use_container_width=True):
        price = float(df["Close"].iloc[-1])
        # use highest-weighted enabled agent as the originator
        best = max(
            (a for a in agents_mod.AGENTS if enabled.get(a, True)),
            key=lambda a: storage.get_agent_stats(a)["weight"],
        )
        learning.close_previous_if_open(sym, price)
        pid = storage.insert_proposal(
            sym, best, "BUY", 0.7, f"Manual from UI (originating agent: {best})", price,
        )
        storage.decide_proposal(pid, "ACCEPT")
        st.toast(f"Manual BUY opened on {sym} via {best}")
        st.rerun()
    if mc2.button("Propose SELL", use_container_width=True):
        price = float(df["Close"].iloc[-1])
        best = max(
            (a for a in agents_mod.AGENTS if enabled.get(a, True)),
            key=lambda a: storage.get_agent_stats(a)["weight"],
        )
        learning.close_previous_if_open(sym, price)
        pid = storage.insert_proposal(
            sym, best, "SELL", 0.7, f"Manual from UI (originating agent: {best})", price,
        )
        storage.decide_proposal(pid, "ACCEPT")
        st.toast(f"Manual SELL opened on {sym} via {best}")
        st.rerun()
    if mc3.button("Close open position", use_container_width=True):
        price = float(df["Close"].iloc[-1])
        learning.close_previous_if_open(sym, price)
        st.toast(f"Closed any open position on {sym}")
        st.rerun()


# ---------------------------------------------------------------------------
# Bottom tabs
# ---------------------------------------------------------------------------
tab_history, tab_agents, tab_params, tab_about = st.tabs(
    ["📜 History", "🏆 Agent leaderboard", "⚙️ Saved parameters", "ℹ️ About"]
)

with tab_history:
    rows = storage.recent_proposals(100)
    if not rows:
        st.info("No proposals yet. Accept/reject a few to see history.")
    else:
        df_hist = pd.DataFrame(rows)
        for col in ("confidence", "price", "realised_pnl"):
            if col in df_hist.columns:
                df_hist[col] = pd.to_numeric(df_hist[col], errors="coerce").round(4)
        # Pretty P&L
        if "realised_pnl" in df_hist.columns:
            df_hist["P&L %"] = df_hist["realised_pnl"].apply(
                lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "open"
            )
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

with tab_agents:
    stats = storage.all_agent_stats()
    if not stats:
        st.info("No agent performance recorded yet.")
    else:
        for s in stats:
            s["weight"] = round(s["weight"], 3)
            s["wins"] = s["wins"]
            s["losses"] = s["losses"]
        df_stats = pd.DataFrame(stats).sort_values("weight", ascending=False)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        st.bar_chart(df_stats.set_index("agent")["weight"])

with tab_params:
    st.json(st.session_state.agent_params)

with tab_about:
    st.markdown(
        """
        **How learning works here**

        - Five independent strategy agents scan the universe daily.
        - Each agent fires a signal with a confidence in `[0, 1]`.
        - The UI surfaces proposals ranked by `confidence × agent_weight`.
        - **You** accept or reject.
        - When a new proposal arrives for a ticker that already has an
          open accepted trade, the old trade closes at the new price and
          the originating agent's weight is updated by the realised return:
            - `weight ← weight * (1 + α · return)`  on a win
            - `weight ← weight / (1 + α · |return|)` on a loss
        - Agents **never** retrain on raw price data — they only learn
          from the closed-loop outcomes of the trades you accepted.
        - Data: Yahoo Finance, 5y history cached on disk, ≥14d online
          window merged on top of the cache at fetch time.
        """
    )
    st.markdown("**Universe (24 instruments)**")
    for grp, items in universe.UNIVERSE.items():
        st.markdown(f"- **{grp}**: " + ", ".join(f"{s} ({n})" for s, n in items))
