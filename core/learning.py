"""
Online learning engine.

Learning does NOT happen from the price series — it happens from the user's
decisions. Every time a previously accepted proposal is closed (a new
proposal arrives for the same ticker), we compute the realised P&L and
update the originating agent's weight.

A simple multi-armed-bandit update:
    weight_new = max(0.05, weight_old + alpha * (realised_return))
"""
from __future__ import annotations

from . import storage


def update_agent_on_close(agent: str, realised_return: float, alpha: float = 0.5) -> None:
    """
    realised_return: signed return, e.g. +0.03 for +3%, -0.02 for -2%.
    Uses a bounded multiplicative step so weights never explode or vanish.
    """
    s = storage.get_agent_stats(agent)
    weight = float(s["weight"])
    wins = int(s["wins"])
    losses = int(s["losses"])

    if realised_return > 0:
        wins += 1
        weight = weight * (1.0 + alpha * realised_return)
    elif realised_return < 0:
        losses += 1
        weight = weight / (1.0 + alpha * abs(realised_return))

    weight = max(0.05, min(weight, 50.0))
    storage.upsert_agent_stats(agent, weight, wins, losses)


def close_previous_if_open(ticker: str, current_price: float) -> None:
    """
    If there's an open accepted proposal for `ticker`, close it at
    `current_price` and apply the realised return to the originating agent.
    """
    open_p = storage.open_accepted_proposal_for(ticker)
    if open_p is None:
        return
    entry_price = float(open_p["price"])
    if entry_price <= 0:
        return
    if open_p["action"] == "BUY":
        realised = (current_price - entry_price) / entry_price
    elif open_p["action"] == "SELL":
        realised = (entry_price - current_price) / entry_price
    else:
        realised = 0.0
    storage.close_open_proposal(open_p["id"], realised)
    update_agent_on_close(open_p["agent"], realised)
