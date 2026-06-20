# Trading Agents — Learning Portfolio

Multiple lightweight strategy agents (no GPU, no neural net) propose
BUY/SELL actions on a 24-instrument universe. The user accepts or
rejects each proposal, and agents learn **only** from the realised
P&L of accepted proposals.

## Features

- 5 tunable strategy agents: SMA Crossover, RSI Mean Reversion,
  Bollinger Breakout, MACD Crossover, Momentum.
- 24-instrument universe: 10 US stocks, 10 EU large-caps, 4 indices.
- 5 years of history from Yahoo Finance, cached to disk.
- ≥14 days of "online" data merged on top of the cache at fetch time.
- Streamlit UI with charts, signals, accept/reject, leaderboard.
- Online learning (multi-armed-bandit style) — weights update only
  from closed-loop outcomes, never from raw price data.
- SQLite for persistence. Zero paid dependencies.

## Run locally

```bash
cd trading_agents
pip install -r requirements.txt
streamlit run ui/app.py
```

The app opens at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud (free)

1. Push the `trading_agents/` directory to a new GitHub repo.
2. Go to <https://share.streamlit.io> and connect the repo.
3. Set **Main file path** to `ui/app.py`.
4. Streamlit Cloud installs `requirements.txt` and exposes a public URL.

Free tier caveat: the app sleeps after ~2 weeks of zero traffic and
wakes on the next request.

## Layout

```
trading_agents/
├── requirements.txt
├── README.md
├── core/
│   ├── universe.py   # 24 tickers grouped by region
│   ├── data.py       # Yahoo Finance fetcher with disk cache
│   ├── indicators.py # SMA / EMA / RSI / Bollinger / MACD / momentum
│   ├── agents.py     # 5 strategy agents + tunable params
│   ├── storage.py    # SQLite tables: proposals, agent_stats, params
│   └── learning.py   # multi-armed-bandit weight updates
├── ui/
│   └── app.py        # Streamlit UI
└── data/             # auto-created; parquet cache + trading.db
```

## Safety

- This app **never** touches a brokerage. It produces proposals only.
- The user clicks Accept/Reject for every trade.
- "Learning" is offline-style: it cannot lose money on its own.
- Past performance of any rule-based strategy is **not** indicative
  of future results.
