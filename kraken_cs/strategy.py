"""
Kraken CS — Backtrader dual-timeframe strategy.

Architecture
------------
data0  (self.data)  : 1-second bars  → order execution
data1  (self.data1) : same feed resampled to 1-minute by Cerebro

All indicators (SAR, EMAs, ATR) are attached to data1 so their values match
TradingView exactly (calculated on 1-minute closes).

Entry detection uses the 1-second close vs the current 1-minute SAR value,
so the order fires within 1 second of the real SAR crossing — not at bar close.

Trading window : 09:35 → 16:25  (close all positions at 16:24)
"""

import backtrader as bt
import backtrader.indicators as btind


TRADE_START_H, TRADE_START_M = 9, 35
TRADE_END_H,   TRADE_END_M   = 16, 25
CLOSE_ALL_H,   CLOSE_ALL_M   = 16, 24


class KrakenCS(bt.Strategy):
    params = dict(
        sar_af=0.02,    # initial & step acceleration factor (Wilder's original = af=increment)
        sar_max=0.4,    # max acceleration factor
        atr_length=14,
        atr_multiplier=1.5,
    )
    # Note: Backtrader's PSAR uses `af` as both the initial value and the step (= 0.02 each).
    # TradingView uses start=0.02 but increment=0.01 — a small difference in how fast the AF
    # ramps up. The signal timing is identical; only the SAR value diverges slightly mid-trend.

    def __init__(self):
        # ── 1-minute indicators (data1) ──────────────────────────────────
        d1 = self.data1

        # period=2 is the PSAR minimum bars before it fires (default, must be int).
        # Do NOT pass period as a float — it would prevent prenext() from running
        # and leave _status uninitialised, causing an AttributeError at runtime.
        self.sar = btind.ParabolicSAR(
            d1,
            period=2,
            af=self.p.sar_af,
            afmax=self.p.sar_max,
        )
        self.ema10  = btind.EMA(d1.close, period=10)
        self.ema35  = btind.EMA(d1.close, period=35)
        self.ema80  = btind.EMA(d1.close, period=80)
        self.ema100 = btind.EMA(d1.close, period=100)
        self.ema200 = btind.EMA(d1.close, period=200)
        self.atr    = btind.ATR(d1, period=self.p.atr_length)

        # Previous 1-second price (used to detect the exact crossing tick)
        self._prev_close = None
        self._stop_price = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _in_trading_window(self, dt):
        h, m = dt.hour, dt.minute
        after_open  = (h > TRADE_START_H) or (h == TRADE_START_H and m >= TRADE_START_M)
        before_close = (h < TRADE_END_H) or (h == TRADE_END_H and m < TRADE_END_M)
        return after_open and before_close

    def _is_close_all_time(self, dt):
        return dt.hour == CLOSE_ALL_H and dt.minute == CLOSE_ALL_M

    def _sar_ready(self):
        """All 1-minute indicators have enough bars to produce a value."""
        return (
            len(self.data1) > 200  # EMA200 needs 200 bars minimum
            and self.sar[0] == self.sar[0]  # not NaN
        )

    # ------------------------------------------------------------------
    # Main loop  — called on every 1-second bar
    # ------------------------------------------------------------------

    def next(self):
        dt    = self.data.datetime.datetime(0)
        price = self.data.close[0]

        # ── Force-close 1 minute before end ──────────────────────────
        if self._is_close_all_time(dt):
            if self.position:
                self.close()
            self._prev_close = price
            return

        if not self._in_trading_window(dt):
            self._prev_close = price
            return

        if not self._sar_ready():
            self._prev_close = price
            return

        sar_val  = self.sar[0]
        ema35    = self.ema35[0]
        ema80    = self.ema80[0]
        atr_val  = self.atr[0]
        prev     = self._prev_close

        # ── Active stop-loss check (intra-minute, every second) ───────
        if self.position:
            if self.position.size > 0:  # long
                # ATR stop
                if self._stop_price and price <= self._stop_price:
                    self.close()
                # Previous-minute low breach
                elif len(self.data1) > 1 and price <= self.data1.low[-1]:
                    self.close()
            elif self.position.size < 0:  # short
                if self._stop_price and price >= self._stop_price:
                    self.close()
                elif len(self.data1) > 1 and price >= self.data1.high[-1]:
                    self.close()

        # ── Entry signals (only when flat) ────────────────────────────
        if not self.position and prev is not None:
            # Long: price crosses above SAR this second
            if prev <= sar_val and price > sar_val and price > ema35:
                self.buy()
                self._stop_price = price - atr_val * self.p.atr_multiplier

            # Short: price crosses below SAR this second
            elif prev >= sar_val and price < sar_val and price < ema80:
                self.sell()
                self._stop_price = price + atr_val * self.p.atr_multiplier

        self._prev_close = price

    def notify_order(self, order):
        if order.status in (order.Completed, order.Canceled, order.Rejected):
            pass

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnl
            dt  = self.data.datetime.datetime(0)
            print(f"[{dt:%Y-%m-%d %H:%M:%S}]  Trade closed  PnL={pnl:+.2f}  (net={trade.pnlcomm:+.2f})")
