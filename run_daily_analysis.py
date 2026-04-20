"""
run_daily_analysis.py
---------------------
Scheduled via Claude Code routine — runs at 09:00 IST every weekday.

What it does:
  1. Discovers top 10 trending Nifty 50 stocks by momentum
  2. Converts the notebook to a Python module on first run (cached after that)
  3. Runs all 5 analysis agents on each stock (no trades are placed here)
  4. Saves daily_recommendations.json to the repo root for run_daily_trades.py

Claude Code routine setup:
  - Trigger  : cron  0 3 * * 1-5   (03:00 UTC = 08:30 IST, before market open)
  - Command  : python run_daily_analysis.py
  - Env vars : see .env.example
"""

import json
import logging
import os
import re
import subprocess
import sys
import importlib.util
from datetime import datetime
from pathlib import Path

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("analysis.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).parent
NOTEBOOK_PATH   = REPO_ROOT / "stock_analyser_agent.ipynb"
MODULE_PATH     = REPO_ROOT / "stock_analyser_agent.py"
OUTPUT_PATH     = REPO_ROOT / "daily_recommendations.json"
CONFIDENCE_THRESHOLD = 80.0


# ── notebook → module conversion ─────────────────────────────────────────────

def _convert_notebook() -> None:
    """Run nbconvert and strip IPython-only lines from the output script."""
    logger.info("Converting notebook to Python module...")
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert",
         "--to", "script", str(NOTEBOOK_PATH),
         "--output", str(MODULE_PATH.stem)],
        check=True,
        capture_output=True,
    )

    # Remove lines that only work inside a notebook kernel
    STRIP_PATTERNS = [
        r"^\s*get_ipython\(",
        r"^\s*%",
        r"^\s*!",
        r"^# In\[",
    ]
    pattern = re.compile("|".join(STRIP_PATTERNS))

    cleaned = []
    for line in MODULE_PATH.read_text(encoding="utf-8").splitlines():
        if not pattern.match(line):
            cleaned.append(line)
    MODULE_PATH.write_text("\n".join(cleaned), encoding="utf-8")
    logger.info("Module written to %s", MODULE_PATH)


def _load_module():
    """Return the stock_analyser_agent module, converting the notebook if needed."""
    if not MODULE_PATH.exists():
        _convert_notebook()

    spec   = importlib.util.spec_from_file_location("stock_analyser_agent", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── config builders ───────────────────────────────────────────────────────────

def _build_api_config(module):
    return module.APIConfig(
        alpha_vantage_key     = os.getenv("ALPHA_VANTAGE_KEY",      ""),
        news_api_key          = os.getenv("NEWS_API_KEY",           ""),
        fred_api_key          = os.getenv("FRED_API_KEY",           ""),
        twitter_bearer_token  = os.getenv("TWITTER_BEARER_TOKEN",   ""),
        reddit_client_id      = os.getenv("REDDIT_CLIENT_ID",       ""),
        reddit_client_secret  = os.getenv("REDDIT_CLIENT_SECRET",   ""),
        openai_api_key        = os.getenv("OPENAI_API_KEY",         ""),
        kite_api_key          = "",   # not needed for analysis
        kite_access_token     = "",
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("Daily Stock Analysis — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    # 1. Discover trending stocks
    from fetch_trending_stocks import fetch_trending_nse_stocks
    symbols = fetch_trending_nse_stocks(n=10)

    if not symbols:
        logger.error("No trending stocks found — aborting.")
        sys.exit(1)

    # 2. Load workflow
    logger.info("Loading analysis workflow...")
    module          = _load_module()
    api_config      = _build_api_config(module)
    analysis_config = module.AnalysisConfig()
    trading_config  = module.TradingConfig(paper_trading=True)   # analysis only

    workflow = module.StockAnalysisWorkflow(api_config, analysis_config, trading_config)

    # 3. Analyse every stock
    logger.info("Analysing %d stocks: %s", len(symbols), symbols)
    raw_results = workflow.analyze_multiple_stocks(symbols)

    # 4. Build structured output
    recommendations = []
    for sym, result in raw_results.items():
        rec = {
            "symbol":           sym,
            "recommendation":   result.get("recommendation",   "HOLD"),
            "confidence_score": result.get("confidence_score", 0.0),
            "current_price":    result.get("current_price",    0.0),
            "price_target":     result.get("price_target",     0.0),
            "stop_loss":        result.get("stop_loss",        0.0),
            "rationale":        result.get("rationale",        ""),
            "errors":           result.get("errors",           []),
        }
        recommendations.append(rec)
        logger.info(
            "  %-20s  %-4s  confidence: %5.1f%%",
            sym, rec["recommendation"], rec["confidence_score"],
        )

    # Sort best opportunities first
    recommendations.sort(key=lambda r: r["confidence_score"], reverse=True)

    actionable = [
        r for r in recommendations
        if r["confidence_score"] >= CONFIDENCE_THRESHOLD
        and r["recommendation"] != "HOLD"
    ]

    output = {
        "date":              datetime.now().strftime("%Y-%m-%d"),
        "generated_at":      datetime.now().isoformat(),
        "stocks_analysed":   len(symbols),
        "recommendations":   recommendations,
        "actionable_trades": actionable,
        "trade_count":       len(actionable),
    }

    # 5. Persist for run_daily_trades.py
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    logger.info("Saved recommendations → %s", OUTPUT_PATH)

    # 6. Summary
    logger.info("-" * 60)
    logger.info("Actionable trades (confidence ≥ %.0f): %d", CONFIDENCE_THRESHOLD, len(actionable))
    for t in actionable:
        logger.info(
            "  → %-4s  %-20s  price: ₹%-8.2f  target: ₹%-8.2f  SL: ₹%.2f",
            t["recommendation"], t["symbol"],
            t["current_price"], t["price_target"], t["stop_loss"],
        )
    logger.info("=" * 60)
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
