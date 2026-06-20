"""
Trading universe: 24 instruments = 10 US stocks + 10 EU stocks + 4 indices.
Ticker format follows Yahoo Finance conventions.
"""

UNIVERSE = {
    "US Tech": [
        ("AAPL", "Apple"),
        ("MSFT", "Microsoft"),
        ("GOOGL", "Alphabet"),
        ("AMZN", "Amazon"),
        ("NVDA", "NVIDIA"),
        ("META", "Meta"),
        ("TSLA", "Tesla"),
    ],
    "US Other": [
        ("BRK-B", "Berkshire Hathaway"),
        ("JPM", "JPMorgan Chase"),
        ("V", "Visa"),
    ],
    "EU Large": [
        ("ASML.AS", "ASML (NL)"),
        ("SAP.DE", "SAP (DE)"),
        ("MC.PA", "LVMH (FR)"),
        ("NOVO-B.CO", "Novo Nordisk (DK)"),
        ("OR.PA", "L'Oréal (FR)"),
        ("SIE.DE", "Siemens (DE)"),
        ("TTE.PA", "TotalEnergies (FR)"),
        ("ITX.MC", "Inditex (ES)"),
        ("AIR.PA", "Airbus (FR)"),
        ("AD.AS", "Ahold Delhaize (NL)"),
    ],
    "Indices": [
        ("^GSPC", "S&P 500"),
        ("^IXIC", "NASDAQ Composite"),
        ("^STOXX50E", "EuroStoxx 50"),
        ("^GDAXI", "DAX 40"),
    ],
}


def all_tickers() -> list[str]:
    """Flatten universe to a list of ticker symbols."""
    out = []
    for group in UNIVERSE.values():
        for sym, _ in group:
            out.append(sym)
    return out


def ticker_name(ticker: str) -> str:
    for group in UNIVERSE.values():
        for sym, name in group:
            if sym == ticker:
                return name
    return ticker
