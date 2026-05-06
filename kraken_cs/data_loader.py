"""
Data loader for Kraken CS backtest.

Supports two sources:
  1. CSV file with columns: datetime, open, high, low, close, volume (1-second bars)
  2. yfinance download (1-minute bars, used when no 1-second data is available)

For true 1-second futures data (ES, NQ, CL), use Interactive Brokers:
  - Install: pip install ib_insync
  - Run IB TWS or Gateway (paper: port 7497, live: port 7496)
  - Call: download_ib_1s_bars(symbol, exchange, start, end)
  - This saves a CSV that can be reloaded with load_csv()
"""

import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, date, timedelta


# ---------------------------------------------------------------------------
# CSV loader  (1-second or 1-minute bars already saved locally)
# ---------------------------------------------------------------------------

def load_csv(filepath: str) -> bt.feeds.PandasData:
    """
    Load OHLCV bars from a CSV file.

    Expected columns (case-insensitive):
        datetime | open | high | low | close | volume

    The datetime column can be any format pandas can parse.
    """
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]

    dt_col = next((c for c in df.columns if "date" in c or "time" in c), df.columns[0])
    df = df.rename(columns={dt_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"CSV is missing required column: '{col}'")

    return bt.feeds.PandasData(dataname=df)


# ---------------------------------------------------------------------------
# yfinance loader  (fallback — 1-minute bars, max 30 days of history)
# ---------------------------------------------------------------------------

def load_yfinance(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1m",
) -> bt.feeds.PandasData:
    """
    Download OHLCV bars from Yahoo Finance.

    ticker   : e.g. "ES=F" (S&P 500 front-month), "NQ=F", "CL=F"
    start    : "YYYY-MM-DD"
    end      : "YYYY-MM-DD"
    interval : "1m" (default) — Yahoo limits history to 30 days for 1m
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker} [{start} → {end}] at {interval}. "
            "Yahoo limits 1-minute history to the last 30 days."
        )

    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df.index.name = "datetime"
    df = df[["open", "high", "low", "close", "volume"]].dropna()

    return bt.feeds.PandasData(dataname=df)


# ---------------------------------------------------------------------------
# Interactive Brokers downloader  (true 1-second futures bars)
# ---------------------------------------------------------------------------

def download_ib_1s_bars(
    symbol: str,
    exchange: str,
    currency: str,
    start: datetime,
    end: datetime,
    output_csv: str,
    port: int = 7497,
) -> str:
    """
    Download 1-second OHLCV bars from Interactive Brokers TWS/Gateway.

    Requires IB TWS or IB Gateway running locally.
      - Paper trading port : 7497
      - Live trading port  : 7496

    symbol     : "ES", "NQ", "CL", etc.
    exchange   : "CME", "NYMEX", etc.
    currency   : "USD"
    start/end  : datetime objects (IB limits 1s bars to ~30 days of history)
    output_csv : path where bars will be saved (reload later with load_csv())
    port       : TWS/Gateway port

    Returns the output_csv path on success.
    """
    try:
        from ib_insync import IB, Future, util
    except ImportError:
        raise ImportError(
            "ib_insync is not installed. Run: pip install ib_insync\n"
            "Also make sure IB TWS or Gateway is running on port " + str(port)
        )

    ib = IB()
    ib.connect("127.0.0.1", port, clientId=1)

    contract = Future(symbol, exchange=exchange, currency=currency)
    ib.qualifyContracts(contract)

    all_bars = []
    cursor = end
    while cursor > start:
        duration_secs = min(int((cursor - start).total_seconds()), 3600)
        if duration_secs <= 0:
            break
        duration_str = f"{duration_secs} S"
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=cursor,
            durationStr=duration_str,
            barSizeSetting="1 secs",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
        )
        if not bars:
            break
        df_chunk = util.df(bars)
        all_bars.append(df_chunk)
        cursor = df_chunk["date"].min()

    ib.disconnect()

    if not all_bars:
        raise RuntimeError("IB returned no bars. Check symbol, exchange, and TWS connection.")

    df = pd.concat(all_bars).drop_duplicates("date").sort_values("date")
    df = df.rename(columns={"date": "datetime"})[["datetime", "open", "high", "low", "close", "volume"]]
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} 1-second bars to {output_csv}")
    return output_csv


# ---------------------------------------------------------------------------
# Synthetic data generator  (for testing / CI without internet access)
# ---------------------------------------------------------------------------

def generate_synthetic_feed(
    start_price: float = 5_200.0,
    days: int = 5,
    bar_interval_seconds: int = 60,
    seed: int = 42,
) -> bt.feeds.PandasData:
    """
    Generate realistic synthetic 1-minute OHLCV bars for testing.

    Produces a random-walk price series with realistic intraday volatility
    and volume patterns, confined to the trading window 09:35-16:25 ET.
    """
    rng = np.random.default_rng(seed)
    bars = []

    base_date = datetime.today().date() - timedelta(days=days + 2)
    price = start_price

    for day_offset in range(days + 2):
        d = base_date + timedelta(days=day_offset)
        if d.weekday() >= 5:  # skip weekends
            continue

        open_dt  = datetime(d.year, d.month, d.day, 9, 35)
        close_dt = datetime(d.year, d.month, d.day, 16, 25)
        current  = open_dt

        while current < close_dt:
            ret   = rng.normal(0, 0.0003)  # ~0.03% per minute
            o     = price
            c     = round(o * (1 + ret), 2)
            h     = round(max(o, c) * (1 + abs(rng.normal(0, 0.0001))), 2)
            l     = round(min(o, c) * (1 - abs(rng.normal(0, 0.0001))), 2)
            vol   = int(rng.integers(200, 800))
            bars.append({"datetime": current, "open": o, "high": h, "low": l, "close": c, "volume": vol})
            price = c
            current += timedelta(seconds=bar_interval_seconds)

    df = pd.DataFrame(bars).set_index("datetime")
    return bt.feeds.PandasData(dataname=df)
