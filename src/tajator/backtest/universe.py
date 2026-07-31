"""A broad, liquid default universe for the edge search — S&P 100 mega-caps plus
index/sector ETFs. Chosen for clean 9-year histories and tight spreads. Dot-class
tickers (e.g. BRK.B) are omitted to avoid IB symbol-formatting special cases.
"""

from __future__ import annotations

MEGACAP_TECH = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "CSCO", "AMD", "INTC", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT", "MU",
]
CONSUMER = [
    "WMT", "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "COST", "PG", "KO", "PEP",
    "PM", "MO", "CL", "MDLZ", "DG",
]
FINANCIALS = [
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB", "PNC",
    "SPGI", "CB", "MMC",
]
HEALTHCARE = [
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "MDT",
]
INDUSTRIALS = [
    "BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "MMM", "UNP", "FDX", "EMR",
]
ENERGY = ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC"]
COMMS = ["DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS"]
OTHER = ["V", "MA", "PYPL", "ACN", "LIN", "NEE", "DUK", "SO"]
ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
    "XLP", "XLU", "XLB", "XLC", "SMH",
]

DEFAULT_UNIVERSE = sorted(set(
    MEGACAP_TECH + CONSUMER + FINANCIALS + HEALTHCARE + INDUSTRIALS
    + ENERGY + COMMS + OTHER + ETFS
))
