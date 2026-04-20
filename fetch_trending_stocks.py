"""
fetch_trending_stocks.py
------------------------
Discovers the top N trending NSE stocks each morning using yfinance.
Scores each Nifty 50 constituent by momentum (price change × volume ratio)
and returns the highest-scoring symbols in SYMBOL.NS format.
"""

import logging
from typing import List, Tuple

import yfinance as yf

logger = logging.getLogger(__name__)

# Nifty 50 universe — extend this list as needed
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS",
    "POWERGRID.NS", "HCLTECH.NS", "MARUTI.NS", "ULTRACEMCO.NS", "TITAN.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "SUNPHARMA.NS", "MM.NS", "TATAMOTORS.NS",
    "NESTLEIND.NS", "TECHM.NS", "ADANIPORTS.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BPCL.NS", "COALINDIA.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS", "SBILIFE.NS", "HDFCLIFE.NS",
    "GRASIM.NS", "ASIANPAINT.NS", "BRITANNIA.NS", "INDUSINDBK.NS", "UPL.NS",
    "SHREECEM.NS", "TATACONSUM.NS", "BAJAJ-AUTO.NS", "LTIM.NS", "HINDALCO.NS",
]


def _score_symbol(symbol: str) -> Tuple[str, float, float]:
    """
    Return (symbol, momentum_score, price_change_pct) for one ticker.
    momentum_score = |price_change_%| × volume_ratio
    Returns score of -1 on failure so the stock sorts to the bottom.
    """
    try:
        hist = yf.Ticker(symbol).history(period="2d")
        if len(hist) < 2:
            return symbol, -1.0, 0.0

        prev_close = hist["Close"].iloc[-2]
        last_close = hist["Close"].iloc[-1]
        prev_vol   = hist["Volume"].iloc[-2]
        last_vol   = hist["Volume"].iloc[-1]

        price_change_pct = (last_close - prev_close) / prev_close * 100
        volume_ratio     = (last_vol / prev_vol) if prev_vol > 0 else 1.0
        momentum_score   = abs(price_change_pct) * volume_ratio

        return symbol, momentum_score, price_change_pct

    except Exception as exc:
        logger.warning("Could not score %s: %s", symbol, exc)
        return symbol, -1.0, 0.0


def fetch_trending_nse_stocks(n: int = 10) -> List[str]:
    """
    Return the top N NSE symbols (e.g. 'RELIANCE.NS') ranked by momentum.
    Only symbols with a positive score are returned.
    """
    logger.info("Scoring %d Nifty 50 stocks for momentum...", len(NIFTY_50))

    scored = [_score_symbol(sym) for sym in NIFTY_50]
    scored = [(sym, score, chg) for sym, score, chg in scored if score > 0]
    scored.sort(key=lambda x: x[1], reverse=True)

    top = scored[:n]
    symbols = [sym for sym, _, _ in top]

    logger.info("Top %d trending stocks today:", n)
    for sym, score, chg in top:
        direction = "▲" if chg >= 0 else "▼"
        logger.info("  %s  %s %.2f%%  (momentum score: %.2f)", sym, direction, chg, score)

    return symbols
