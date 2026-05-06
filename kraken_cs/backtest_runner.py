"""
Kraken CS — Backtest runner.

Usage
-----
    python backtest_runner.py

Priority order for data:
  1. Local CSV (set USE_CSV = True)           ← best: real 1-second IB bars
  2. yfinance download (1-minute, max 30 days) ← good: free, works on your machine
  3. Synthetic data                            ← fallback for offline testing only

Key Cerebro setup
-----------------
  cerebro.adddata(data_1s)                              → data0 : raw bars (execution)
  cerebro.resampledata(data_1s_copy, timeframe=Minutes) → data1 : 1-minute bars (indicators)

The strategy reads data1 for SAR/EMA/ATR (identical to TradingView 1-min chart) and
data0 for the intra-bar price movement, so entries fire the second the SAR is crossed —
not at bar close.
"""

import os
import datetime

# Force non-GUI backend before ANY matplotlib import so it works in headless environments.
# Remove or comment these two lines if you want an interactive chart window on your machine.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

import backtrader as bt
import backtrader.analyzers as btanalyzers

from data_loader import load_yfinance, load_csv, generate_synthetic_feed
from strategy import KrakenCS


# =============================================================================
# CONFIG  —  edit these to match your instrument and data
# =============================================================================

# ── Data source ──────────────────────────────────────────────────────────────

# Option 1: Real 1-second bars from IB (saved to CSV with data_loader.download_ib_1s_bars)
USE_CSV  = False
CSV_PATH = "data/es_1sec.csv"

# Option 2: yfinance  (1-minute bars, limited to last 30 days)
TICKER     = "ES=F"       # ES=F (S&P 500), NQ=F (Nasdaq), CL=F (Crude Oil)
END_DATE   = datetime.date.today().strftime("%Y-%m-%d")
START_DATE = (datetime.date.today() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

# ── Account settings ─────────────────────────────────────────────────────────

INITIAL_CASH  = 100_000.0
COMMISSION    = 2.05        # per-side in $ (typical ES micro = $0.62, full = $2.05)
SLIPPAGE_PERC = 0.0001      # 0.01% per fill


# =============================================================================
# MAIN
# =============================================================================

def _load_data():
    """Return a Backtrader data feed, trying sources in priority order."""
    if USE_CSV:
        print(f"Loading CSV: {CSV_PATH}")
        return load_csv(CSV_PATH)

    print(f"Downloading {TICKER}  [{START_DATE} → {END_DATE}]  via yfinance …")
    try:
        feed = load_yfinance(TICKER, start=START_DATE, end=END_DATE, interval="1m")
        print("Download complete.")
        return feed
    except Exception as e:
        print(f"yfinance failed ({e})\nFalling back to synthetic data for testing.")
        return generate_synthetic_feed(days=5)


def run():
    cerebro = bt.Cerebro()

    # ── Data ─────────────────────────────────────────────────────────────────
    # Backtrader's resampledata() consumes the feed object it receives, so we
    # need two separate feed instances pointing at the same underlying data.
    feed_exec  = _load_data()   # data0 — execution (raw bar resolution)
    feed_indic = _load_data()   # data1 — indicators (resampled to 1-minute)

    cerebro.adddata(feed_exec, name="exec")

    # When raw data is already 1-minute, resampledata is a no-op for the
    # 1-minute resampling but keeps the dual-feed contract intact — swapping
    # in true 1-second bars later requires zero strategy changes.
    cerebro.resampledata(
        feed_indic,
        name="indicators_1min",
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
    )

    # ── Strategy ─────────────────────────────────────────────────────────────
    cerebro.addstrategy(KrakenCS)

    # ── Broker ───────────────────────────────────────────────────────────────
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION, commtype=bt.CommInfoBase.COMM_FIXED)
    cerebro.broker.set_slippage_perc(SLIPPAGE_PERC)

    # ── Analyzers ────────────────────────────────────────────────────────────
    cerebro.addanalyzer(btanalyzers.SharpeRatio,   _name="sharpe",   riskfreerate=0.04)
    cerebro.addanalyzer(btanalyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(btanalyzers.Returns,       _name="returns")

    # ── Run ──────────────────────────────────────────────────────────────────
    print(f"\nStarting portfolio : ${cerebro.broker.getvalue():,.2f}")
    results = cerebro.run()
    strat   = results[0]
    print(f"Ending portfolio   : ${cerebro.broker.getvalue():,.2f}\n")

    _print_report(strat)

    try:
        import matplotlib.pyplot as plt
        figs = cerebro.plot(style="candlestick", volume=True, iplot=False)
        chart_path = "backtest_chart.png"
        figs[0][0].savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"\nChart saved → {chart_path}")
    except Exception as e:
        print(f"\n(Chart skipped — {e})")


def _print_report(strat):
    print("=" * 55)
    print("  KRAKEN CS — BACKTEST RESULTS")
    print("=" * 55)

    sharpe = strat.analyzers.sharpe.get_analysis()
    sr = sharpe.get("sharperatio", None)
    print(f"  Sharpe Ratio     : {sr:.4f}" if sr else "  Sharpe Ratio     : n/a")

    dd = strat.analyzers.drawdown.get_analysis()
    print(f"  Max Drawdown     : {dd.max.drawdown:.2f}%")
    print(f"  Max DD $ Amount  : ${dd.max.moneydown:,.2f}")

    ta = strat.analyzers.trades.get_analysis()
    total   = ta.get("total",  {}).get("total",   0)
    won     = ta.get("won",    {}).get("total",   0)
    lost    = ta.get("lost",   {}).get("total",   0)
    pnl_net = ta.get("pnl",   {}).get("net",      {}).get("total", 0)
    win_pct = (won / total * 100) if total else 0
    avg_won  = ta.get("won",  {}).get("pnl", {}).get("average", 0)
    avg_lost = ta.get("lost", {}).get("pnl", {}).get("average", 0)

    print(f"  Total Trades     : {total}")
    print(f"  Won / Lost       : {won} / {lost}  ({win_pct:.1f}% win rate)")
    print(f"  Net P&L          : ${pnl_net:,.2f}")
    print(f"  Avg Win          : ${avg_won:,.2f}")
    print(f"  Avg Loss         : ${avg_lost:,.2f}")

    ret  = strat.analyzers.returns.get_analysis()
    rtot = ret.get("rtot", 0)
    print(f"  Total Return     : {rtot * 100:.2f}%")

    print("=" * 55)


if __name__ == "__main__":
    run()
