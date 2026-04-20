"""
run_daily_trades.py
-------------------
Run this LOCALLY each morning after refreshing your Kite Connect access token.

What it does:
  1. Reads daily_recommendations.json written by the Claude Code routine
  2. Shows you the actionable trades and asks for confirmation
  3. Places live orders on NSE via Kite Connect for recommendations with
     confidence >= 80 and recommendation != HOLD

Usage:
  # Option A — pass token as env var
  set KITE_ACCESS_TOKEN=your_token_here
  python run_daily_trades.py

  # Option B — script will prompt you
  python run_daily_trades.py

  # Dry-run (paper mode, no real orders)
  python run_daily_trades.py --paper
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from kiteconnect import KiteConnect

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trades.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

REPO_ROOT             = Path(__file__).parent
RECOMMENDATIONS_FILE  = REPO_ROOT / "daily_recommendations.json"
CONFIDENCE_THRESHOLD  = 80.0


# ── Kite Connect helpers ──────────────────────────────────────────────────────

def _clean_symbol(symbol: str) -> str:
    """Strip exchange suffix: RELIANCE.NS → RELIANCE."""
    return symbol.split(".")[0]


def _place_order(kite: KiteConnect, trade: dict, trading_cfg: dict, paper: bool) -> dict:
    symbol       = _clean_symbol(trade["symbol"])
    txn_type     = trade["recommendation"]          # "BUY" or "SELL"
    current_price = trade["current_price"]
    max_value    = trading_cfg["max_order_value"]
    quantity     = max(1, int(max_value / current_price)) if current_price > 0 else 1

    if paper:
        order_id = f"PAPER-{symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.info(
            "[PAPER] %-4s  %d × %-15s  @ ₹%.2f  (value: ₹%.2f)  SL: ₹%.2f  target: ₹%.2f",
            txn_type, quantity, symbol, current_price,
            quantity * current_price, trade["stop_loss"], trade["price_target"],
        )
        return {"executed": True, "paper": True, "order_id": order_id,
                "symbol": symbol, "quantity": quantity, "status": "PAPER_COMPLETE"}

    try:
        kite_txn = (
            kite.TRANSACTION_TYPE_BUY
            if txn_type == "BUY"
            else kite.TRANSACTION_TYPE_SELL
        )
        order_type = (
            kite.ORDER_TYPE_MARKET
            if trading_cfg["order_type"] == "MARKET"
            else kite.ORDER_TYPE_LIMIT
        )
        product = (
            kite.PRODUCT_CNC
            if trading_cfg["product"] == "CNC"
            else kite.PRODUCT_MIS
        )

        order_id = kite.place_order(
            variety          = kite.VARIETY_REGULAR,
            tradingsymbol    = symbol,
            exchange         = trading_cfg["exchange"],
            transaction_type = kite_txn,
            quantity         = quantity,
            order_type       = order_type,
            product          = product,
        )

        logger.info(
            "[LIVE]  %-4s  %d × %-15s  order_id: %s",
            txn_type, quantity, symbol, order_id,
        )
        return {"executed": True, "paper": False, "order_id": str(order_id),
                "symbol": symbol, "quantity": quantity, "status": "ORDER_PLACED"}

    except Exception as exc:
        logger.error("Order failed for %s: %s", symbol, exc)
        return {"executed": False, "paper": False, "symbol": symbol,
                "error": str(exc), "status": "FAILED"}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true",
                        help="Simulate trades without placing real orders")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Daily Trade Execution — %s  [%s]",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "PAPER" if args.paper else "LIVE")
    logger.info("=" * 60)

    # 1. Load recommendations
    if not RECOMMENDATIONS_FILE.exists():
        logger.error("daily_recommendations.json not found.")
        logger.error("Make sure the Claude Code routine ran successfully first.")
        sys.exit(1)

    data = json.loads(RECOMMENDATIONS_FILE.read_text(encoding="utf-8"))
    today = datetime.now().strftime("%Y-%m-%d")

    if data.get("date") != today:
        logger.warning(
            "Recommendations are from %s, not today (%s).",
            data.get("date"), today
        )
        answer = input("Continue anyway? (yes/no): ").strip().lower()
        if answer != "yes":
            sys.exit(0)

    actionable = data.get("actionable_trades", [])
    if not actionable:
        logger.info("No actionable trades today (nothing with confidence ≥ %.0f).", CONFIDENCE_THRESHOLD)
        return

    # 2. Show trades and confirm
    logger.info("Actionable trades:")
    for t in actionable:
        logger.info(
            "  %-4s  %-20s  confidence: %5.1f%%  price: ₹%.2f  target: ₹%.2f  SL: ₹%.2f",
            t["recommendation"], t["symbol"], t["confidence_score"],
            t["current_price"], t["price_target"], t["stop_loss"],
        )

    mode_label = "paper-trade" if args.paper else "place LIVE orders for"
    answer = input(f"\n{mode_label.capitalize()} {len(actionable)} trade(s) on NSE? (yes/no): ").strip().lower()
    if answer != "yes":
        logger.info("Aborted by user.")
        sys.exit(0)

    # 3. Get Kite credentials (skip for paper mode)
    kite = None
    if not args.paper:
        api_key = os.getenv("KITE_API_KEY") or input("Kite API key: ").strip()
        access_token = os.getenv("KITE_ACCESS_TOKEN") or input("Kite access token: ").strip()

        if not api_key or not access_token:
            logger.error("API key and access token are required for live trading.")
            sys.exit(1)

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        logger.info("Kite Connect authenticated.")

    # 4. Trading config from env vars (with sensible defaults)
    trading_cfg = {
        "exchange":        os.getenv("EXCHANGE",        "NSE"),
        "order_type":      os.getenv("ORDER_TYPE",      "MARKET"),
        "product":         os.getenv("PRODUCT",         "CNC"),
        "max_order_value": float(os.getenv("MAX_ORDER_VALUE", "50000")),
    }

    # 5. Place orders
    results = []
    for trade in actionable:
        result = _place_order(kite, trade, trading_cfg, paper=args.paper)
        result["recommendation"] = trade["recommendation"]
        result["confidence_score"] = trade["confidence_score"]
        results.append(result)

    # 6. Summary
    succeeded = sum(1 for r in results if r["executed"])
    failed    = len(results) - succeeded
    logger.info("-" * 60)
    logger.info("Orders placed: %d  |  Failed: %d", succeeded, failed)

    # 7. Save trade log
    log_file = REPO_ROOT / f"trade_log_{today}.json"
    log_file.write_text(
        json.dumps({"date": today, "paper": args.paper, "trades": results}, indent=2),
        encoding="utf-8",
    )
    logger.info("Trade log saved → %s", log_file)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
