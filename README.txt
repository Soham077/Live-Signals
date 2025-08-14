RISK & BLUNT WARNINGS
- 7-day history = low statistical power. MC helps but does not replace real multi-month live data.
- RISK_PER_TRADE = 1% can create big equity swings; do not scale until you have ~200 real trades or MC shows >70% profitable trials AND avg trades per trial ≥ 200.
- Realtime signals = signals only. Execution slippage, partial fills, API failures will change real PnL.

DEPENDENCIES
- Python 3.9+
- pip packages:
  pip install pandas numpy matplotlib tqdm websocket-client ta python-binance

REQUIRED INPUT FILES (preferred for backtest + MC)
- hist_btc.csv, hist_xau.csv, hist_eurusd.csv, hist_spx.csv
- Required columns (case-sensitive): datetime,open,high,low,close,volume
- Datetime: ISO8601 with timezone OR UTC. If your files are in IST/Kolkata, set --timezone_override Asia/Kolkata.
- Each CSV should contain EXACTLY 7 contiguous days of the chosen timeframe (default 5m). If not, script will warn and still run with low confidence.

DEFAULTS
- TIMEFRAMES: 1m,5m,15m,1h (primary=5m)
- INSTRUMENTS: BTC-USD, XAU-USD, EUR-USD, SPX
- INITIAL_CAPITAL=$1000 per instrument
- RISK_PER_TRADE=1% (0.01)
- MC_TRIALS=1000 per instrument (block-bootstrap by day)
- SLIPPAGE_SCENARIOS=[0.0005,0.001,0.005]
- COMMISSION_PCT=0.0005 per side
- ALLOW_SHORTS=False (enable with --allow_shorts)

RUN
- Backtest+MC for all instruments:
  python code_used_multi.py --mode backtest --csv_btc hist_btc.csv --csv_xau hist_xau.csv --csv_fx hist_eurusd.csv --csv_spx hist_spx.csv --mc 1000 --risk 0.01

- Realtime (default feeds: Binance BTC websocket; others poll local rolling CSV if provided):
  python code_used_multi.py --mode realtime

- Both (backtest+MC then start realtime):
  python code_used_multi.py --mode both

OUTPUTS
Per instrument:
1. signals_backtest_<instrument>.csv — per trade backtest signal log
2. trades_backtest_<instrument>.csv — same as signals
3. mc_summary_<instrument>.json — MC percentiles and stats per slippage & param combo
4. bt_equity_curve_<instrument>.png, bt_drawdown_<instrument>.png, mc_returns_hist_<instrument>_<slippage>.png

Realtime outputs:
- realtime_signals.csv — appended: timestamp,instrument,timeframe,side,price,size,stop,target,confidence_pct,reason_text
- realtime_dashboard.json — rolling status (current indicators, TF alignment, last signal)

Global:
- combined_actionability.txt — summary table across instruments (Actionability score 0–10) and blunt recommendation
- code_used_multi.py — this script
- README.txt — this file

LIVE FEEDS
- BTC: Binance websocket (btcusdt@kline_1m/5m/15m/1h) is enabled by default if websocket-client is installed.
- XAU/EURUSD/SPX: realtime expects a local rolling CSV (append new candles). You can also provide REST polling by updating the CSV path to a file populated by your poller.

NOTES
- SPX session anchor uses US market hours (09:30–16:00 ET) for VWAP sessions; FX/CRYPTO use UTC day sessions.
- If you want to trade SPX futures or enable shorts on SPX, modify allow_shorts and supply symbol mapping. This script emits signals only; no live orders are placed.