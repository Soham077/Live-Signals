import argparse
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

try:
	from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
	ZoneInfo = None

# Optional: websocket for realtime
try:
	from websocket import WebSocketApp
except Exception:  # pragma: no cover
	WebSocketApp = None

# Optional: TA indicators
try:
	import ta
except Exception:  # pragma: no cover
	ta = None

RISK_WARNING_HEADER = (
	"RISK & BLUNT WARNINGS\n"
	"- 7-day history = low statistical power. MC helps but does not replace real multi-month live data.\n"
	"- RISK_PER_TRADE = 1% can create big equity swings; do not scale until you have ~200 real trades or MC shows >70% profitable trials AND avg trades per trial ≥ 200.\n"
	"- Realtime signals = signals only. Execution slippage, partial fills, API failures will change real PnL.\n\n"
)

DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h"]
PRIMARY_TF = "5m"

# DEFAULTS/OVERRIDES
DEFAULT_INITIAL_CAPITAL = 1000.0
DEFAULT_RISK_PER_TRADE = 0.01
DEFAULT_MC_TRIALS = 1000
DEFAULT_SLIPPAGE_SCENARIOS = [0.0005, 0.001, 0.005]
DEFAULT_COMMISSION_PCT = 0.0005
DEFAULT_ATR_LEN = 20
DEFAULT_EMA_LEN = 20
DEFAULT_VOL_MA_LEN = 20
DEFAULT_VOL_MULT = 1.25
DEFAULT_ATR_MULT_FOR_STOP = 1.0
DEFAULT_TARGET_MULT = 2.0
DEFAULT_MIN_TRADES_CONFIDENCE = 200
DEFAULT_ALLOW_SHORTS = False

PARAM_SWEEP_ATR_MULT = [0.8, 1.0, 1.2]
PARAM_SWEEP_VOL_MULT = [1.0, 1.25, 1.5]
PARAM_SWEEP_RISK = [0.005, 0.01, 0.02]

INSTRUMENTS = ["BTC-USD", "XAU-USD", "EUR-USD", "SPX"]

BINANCE_WS_URL = (
	"wss://stream.binance.com:9443/stream?streams="
	"btcusdt@kline_1m/"
	"btcusdt@kline_5m/"
	"btcusdt@kline_15m/"
	"btcusdt@kline_1h"
)

@dataclass
class StrategyParams:
	atr_len: int = DEFAULT_ATR_LEN
	ema_len: int = DEFAULT_EMA_LEN
	vol_ma_len: int = DEFAULT_VOL_MA_LEN
	vol_mult: float = DEFAULT_VOL_MULT
	atr_mult_for_stop: float = DEFAULT_ATR_MULT_FOR_STOP
	target_mult: float = DEFAULT_TARGET_MULT
	risk_per_trade: float = DEFAULT_RISK_PER_TRADE
	commission_pct: float = DEFAULT_COMMISSION_PCT
	slippage_pct: float = DEFAULT_SLIPPAGE_SCENARIOS[0]
	allow_shorts: bool = DEFAULT_ALLOW_SHORTS

@dataclass
class TradeRecord:
	entry_time: pd.Timestamp
	side: str  # "long" or "short"
	entry_price: float
	size: float
	stop: float
	target: float
	exit_time: Optional[pd.Timestamp]
	exit_price: Optional[float]
	pnl_usd: Optional[float]
	pnl_pct: Optional[float]
	holding_minutes: Optional[int]
	exit_reason: Optional[str]
	reason_text: str

@dataclass
class BacktestResult:
	trades: List[TradeRecord]
	equity_curve: pd.Series
	drawdown_curve: pd.Series
	stats: Dict[str, float]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Multi-instrument real-time signal generator + backtest + MC robustness"
	)
	parser.add_argument("--mode", choices=["backtest", "realtime", "both"], default="both")
	parser.add_argument("--csv_btc", type=str, default="hist_btc.csv")
	parser.add_argument("--csv_xau", type=str, default="hist_xau.csv")
	parser.add_argument("--csv_fx", type=str, default="hist_eurusd.csv")
	parser.add_argument("--csv_spx", type=str, default="hist_spx.csv")
	parser.add_argument("--initial_capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
	parser.add_argument("--risk", type=float, default=DEFAULT_RISK_PER_TRADE)
	parser.add_argument("--mc", type=int, default=DEFAULT_MC_TRIALS)
	parser.add_argument("--timeframes", type=str, default=",".join(DEFAULT_TIMEFRAMES), help="Comma-separated TFs: e.g., 1m,5m,15m,1h")
	parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE_SCENARIOS[0])
	parser.add_argument("--allow_shorts", action="store_true", default=DEFAULT_ALLOW_SHORTS)
	parser.add_argument("--outdir", type=str, default=".")
	parser.add_argument("--timezone_override", type=str, default="UTC", help="If CSVs use a specific timezone like Asia/Kolkata, set this.")
	parser.add_argument("--quiet", action="store_true")
	return parser.parse_args()


# ------------------------ IO HELPERS ------------------------

def ensure_datetime_index(df: pd.DataFrame, timezone_name: str = "UTC") -> pd.DataFrame:
	if "datetime" not in df.columns:
		raise ValueError("CSV must include 'datetime' column")
	df = df.copy()
	df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
	if df["datetime"].isna().any():
		raise ValueError("Failed to parse some datetime values; ensure ISO8601 or UTC timestamps")
	df = df.set_index("datetime").sort_index()
	# Localize/convert if override provided (assuming input is UTC already or timezone-aware)
	if timezone_name and timezone_name.upper() != "UTC" and ZoneInfo is not None:
		try:
			local_tz = ZoneInfo(timezone_name)
			# Convert to target tz then back to UTC for consistency; keep UTC index
			df = df.tz_convert(local_tz).tz_convert("UTC")
		except Exception:
			pass
	return df


def read_history_csv(path: str, timezone_name: str = "UTC") -> Optional[pd.DataFrame]:
	if not os.path.exists(path):
		return None
	df = pd.read_csv(path)
	required = ["open", "high", "low", "close", "volume", "datetime"]
	for col in required:
		if col not in df.columns:
			raise ValueError(f"CSV {path} missing required column: {col}")
	df = ensure_datetime_index(df, timezone_name)
	return df


# ------------------------ TIMEFRAME UTILS ------------------------

def timeframe_to_pandas_rule(tf: str) -> str:
	mapping = {"1m": "1T", "5m": "5T", "15m": "15T", "1h": "1H"}
	if tf not in mapping:
		raise ValueError(f"Unsupported timeframe: {tf}")
	return mapping[tf]


def resample_ohlcv(df: pd.DataFrame, tf: str) -> pd.DataFrame:
	if df.empty:
		return df.copy()
	rule = timeframe_to_pandas_rule(tf)
	agg = {
		"open": "first",
		"high": "max",
		"low": "min",
		"close": "last",
		"volume": "sum",
	}
	res = df.resample(rule, label="right", closed="right").apply(agg).dropna()
	return res


# ------------------------ SESSION / VWAP ------------------------

def get_instrument_type(instrument: str) -> str:
	instrument_upper = instrument.upper()
	if instrument_upper == "BTC-USD":
		return "CRYPTO"
	if instrument_upper in ("XAU-USD", "EUR-USD"):
		return "FX"
	if instrument_upper == "SPX":
		return "EQUITY"
	return "UNKNOWN"


def session_id_for_row(ts_utc: pd.Timestamp, instrument: str) -> str:
	inst_type = get_instrument_type(instrument)
	if inst_type in ("CRYPTO", "FX"):
		# UTC day
		return ts_utc.tz_convert("UTC").strftime("%Y-%m-%d")
	elif inst_type == "EQUITY":
		# US market session 09:30–16:00 ET
		if ZoneInfo is None:
			return ts_utc.tz_convert("UTC").strftime("%Y-%m-%d")
		et = ts_utc.tz_convert(ZoneInfo("America/New_York"))
		# If outside RTH, map to date of session boundary by candle timestamp date
		return et.strftime("%Y-%m-%d")
	else:
		return ts_utc.tz_convert("UTC").strftime("%Y-%m-%d")


def add_session_vwap(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
	if df.empty:
		return df.copy()
	df = df.copy()
	if df.index.tz is None:
		df.index = df.index.tz_localize("UTC")
	# Typical price for VWAP
	if not all(c in df.columns for c in ["high", "low", "close"]):
		raise ValueError("DataFrame must have high, low, close for VWAP")
	df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
	if "volume" not in df.columns:
		df["volume"] = 1.0
	# Compute session id per row
	df["session_id"] = [session_id_for_row(ts, instrument) for ts in df.index]
	# Cumulative PV and V per session
	pv = df["typical_price"] * df["volume"]
	df["cum_pv"] = pv.groupby(df["session_id"]).cumsum()
	df["cum_v"] = df["volume"].groupby(df["session_id"]).cumsum()
	df["vwap"] = df["cum_pv"] / df["cum_v"].replace(0.0, np.nan)
	return df


# ------------------------ TECHNICAL INDICATORS ------------------------

def add_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
	if df.empty:
		return df.copy()
	df = df.copy()
	# EMA20
	if ta is not None:
		try:
			df["ema20"] = ta.trend.EMAIndicator(close=df["close"], window=params.ema_len).ema_indicator()
		except Exception:
			df["ema20"] = df["close"].ewm(span=params.ema_len, adjust=False).mean()
	else:
		df["ema20"] = df["close"].ewm(span=params.ema_len, adjust=False).mean()
	# ATR
	if ta is not None:
		try:
			atr_ind = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=params.atr_len)
			df["atr"] = atr_ind.average_true_range()
		except Exception:
			hl = df["high"] - df["low"]
			pc = (df["close"] - df["close"].shift(1)).abs()
			df["atr"] = pd.concat([hl, pc], axis=1).max(axis=1).rolling(params.atr_len).mean()
	else:
		hl = df["high"] - df["low"]
		pc = (df["close"] - df["close"].shift(1)).abs()
		df["atr"] = pd.concat([hl, pc], axis=1).max(axis=1).rolling(params.atr_len).mean()
	# MACD histogram 12/26/9
	if ta is not None:
		try:
			macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
			df["macd_hist"] = macd.macd_diff()
			df["macd"] = macd.macd()
			df["macd_signal"] = macd.macd_signal()
		except Exception:
			df["macd"], df["macd_signal"], df["macd_hist"] = _macd_fallback(df["close"])  # type: ignore
	else:
		df["macd"], df["macd_signal"], df["macd_hist"] = _macd_fallback(df["close"])  # type: ignore
	# Volume MA
	if "volume" not in df.columns:
		df["volume"] = 1.0
	df["vol_ma"] = df["volume"].rolling(params.vol_ma_len).mean()
	return df


def _macd_fallback(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
	ema_fast = close.ewm(span=fast, adjust=False).mean()
	ema_slow = close.ewm(span=slow, adjust=False).mean()
	macd = ema_fast - ema_slow
	macd_signal = macd.ewm(span=signal, adjust=False).mean()
	macd_hist = macd - macd_signal
	return macd, macd_signal, macd_hist


# ------------------------ STRATEGY LOGIC ------------------------

def compute_confidence(primary_ok: bool,
					  vol_multiplier_hit: bool,
					  macd_hist_value: float,
					  macd_hist_top_quartile_threshold: float,
					  trend_15m_ok: bool,
					  trend_1h_ok: bool,
					  higher_tf_disagree: bool,
					  mc_median_return_leq_zero: bool) -> int:
	confidence = 50
	if primary_ok:
		confidence += 15
	if trend_15m_ok:
		confidence += 10
	if trend_1h_ok:
		confidence += 10
	if abs(macd_hist_value) >= macd_hist_top_quartile_threshold and macd_hist_value > 0:
		confidence += 10
	if vol_multiplier_hit:
		confidence += 5
	if higher_tf_disagree:
		confidence -= 20
	if mc_median_return_leq_zero:
		confidence -= 10
	return int(max(0, min(100, confidence)))


def vwap_flip_long_condition(df5: pd.DataFrame, idx: pd.Timestamp) -> bool:
	"""True if close > vwap and either previous close <= vwap or a retest (low <= vwap) then close above."""
	if idx not in df5.index:
		return False
	row = df5.loc[idx]
	if pd.isna(row.get("vwap")):
		return False
	close_now = row["close"]
	vwap_now = row["vwap"]
	if close_now <= vwap_now:
		return False
	prev_idx = df5.index.get_loc(idx) - 1
	if prev_idx < 0:
		return False
	prev_row = df5.iloc[prev_idx]
	prev_close = prev_row["close"]
	prev_vwap = prev_row["vwap"]
	# Flip or retest
	if prev_close <= prev_vwap:
		return True
	# Retest within candle
	low_now = row["low"]
	if low_now <= vwap_now:
		return True
	return False


def vwap_flip_short_condition(df5: pd.DataFrame, idx: pd.Timestamp) -> bool:
	if idx not in df5.index:
		return False
	row = df5.loc[idx]
	if pd.isna(row.get("vwap")):
		return False
	close_now = row["close"]
	vwap_now = row["vwap"]
	if close_now >= vwap_now:
		return False
	prev_idx = df5.index.get_loc(idx) - 1
	if prev_idx < 0:
		return False
	prev_row = df5.iloc[prev_idx]
	prev_close = prev_row["close"]
	prev_vwap = prev_row["vwap"]
	if prev_close >= prev_vwap:
		return True
	high_now = row["high"]
	if high_now >= vwap_now:
		return True
	return False


def higher_tf_alignment(df15: Optional[pd.DataFrame], df1h: Optional[pd.DataFrame], ts: pd.Timestamp, direction: str) -> Tuple[bool, bool, bool]:
	"""Return (trend_15_ok, trend_1h_ok, higher_tf_disagree)."""
	trend_15 = False
	trend_1h = False
	if df15 is not None and not df15.empty and ts in df15.index:
		row15 = df15.loc[ts]
		macd_ok_15 = row15.get("macd_hist", 0.0) > 0
		if direction == "long":
			trend_15 = (row15.get("close", np.nan) > row15.get("ema20", np.nan)) and macd_ok_15
		else:
			trend_15 = (row15.get("close", np.nan) < row15.get("ema20", np.nan)) and (row15.get("macd_hist", 0.0) < 0)
	if df1h is not None and not df1h.empty and ts in df1h.index:
		row1h = df1h.loc[ts]
		macd_ok_1h = row1h.get("macd_hist", 0.0) > 0
		if direction == "long":
			trend_1h = (row1h.get("close", np.nan) > row1h.get("ema20", np.nan)) and macd_ok_1h
		else:
			trend_1h = (row1h.get("close", np.nan) < row1h.get("ema20", np.nan)) and (row1h.get("macd_hist", 0.0) < 0)
	# Disagree if at least one exists and it is opposite to direction
	disagree = False
	if direction == "long":
		if df15 is not None and ts in df15.index and (df15.loc[ts].get("close", np.nan) < df15.loc[ts].get("ema20", np.nan)):
			disagree = True
		if df1h is not None and ts in df1h.index and (df1h.loc[ts].get("close", np.nan) < df1h.loc[ts].get("ema20", np.nan)):
			disagree = True
	else:
		if df15 is not None and ts in df15.index and (df15.loc[ts].get("close", np.nan) > df15.loc[ts].get("ema20", np.nan)):
			disagree = True
		if df1h is not None and ts in df1h.index and (df1h.loc[ts].get("close", np.nan) > df1h.loc[ts].get("ema20", np.nan)):
			disagree = True
	return trend_15, trend_1h, disagree


# ------------------------ BACKTEST ENGINE ------------------------

def simulate_trades(
	instrument: str,
	df5: pd.DataFrame,
	df15: Optional[pd.DataFrame],
	df1h: Optional[pd.DataFrame],
	params: StrategyParams,
	initial_capital: float,
	quiet: bool = False,
) -> BacktestResult:
	"""Run backtest on primary 5m timeframe with multi-TF confirmation."""
	if df5.empty:
		return BacktestResult(trades=[], equity_curve=pd.Series(dtype=float), drawdown_curve=pd.Series(dtype=float), stats={})

	df5 = df5.copy()
	# Precompute empirical threshold for MACD hist magnitude top quartile over period
	macd_hist_abs = df5["macd_hist"].abs().dropna()
	macd_top_quartile = float(np.percentile(macd_hist_abs, 75)) if not macd_hist_abs.empty else 0.0

	equity = initial_capital
	trades: List[TradeRecord] = []
	open_position: Optional[TradeRecord] = None
	equity_curve = []
	curve_index = []

	for i in range(1, len(df5.index)):
		idx = df5.index[i]
		row = df5.iloc[i]
		prev_row = df5.iloc[i - 1]

		# Update open position with trailing/stop/target logic using current candle
		if open_position is not None:
			atr_now = float(row.get("atr", np.nan))
			# Simulate intra-candle hit: stop or target
			if open_position.side == "long":
				stop_hit = (row["low"] <= open_position.stop)
				target_hit = (row["high"] >= open_position.target)
				# Trail after 1R reached
				if row["high"] >= open_position.entry_price + (open_position.entry_price - open_position.stop):
					trail = row["close"] - params.atr_mult_for_stop * atr_now
					open_position.stop = max(open_position.stop, trail)
				if stop_hit or target_hit:
					exit_price = open_position.stop if stop_hit and not target_hit else open_position.target
					exit_price = float(exit_price)
					exit_price *= (1.0 - params.slippage_pct)
					commission = (abs(open_position.size) * exit_price) * params.commission_pct
					pnl = (exit_price - open_position.entry_price) * open_position.size - commission
					equity += pnl
					tr_minutes = int((idx - open_position.entry_time).total_seconds() // 60)
					exit_reason = "stop" if stop_hit and not target_hit else ("target" if target_hit and not stop_hit else "stop_target_same_candle")
					open_position.exit_time = idx
					open_position.exit_price = exit_price
					open_position.pnl_usd = pnl
					open_position.pnl_pct = pnl / max(1e-9, initial_capital)
					open_position.holding_minutes = tr_minutes
					open_position.exit_reason = exit_reason
					trades.append(open_position)
					open_position = None
			elif open_position.side == "short":
				stop_hit = (row["high"] >= open_position.stop)
				target_hit = (row["low"] <= open_position.target)
				if row["low"] <= open_position.entry_price - (open_position.stop - open_position.entry_price):
					trail = row["close"] + params.atr_mult_for_stop * atr_now
					open_position.stop = min(open_position.stop, trail)
				if stop_hit or target_hit:
					exit_price = open_position.stop if stop_hit and not target_hit else open_position.target
					exit_price = float(exit_price)
					exit_price *= (1.0 + params.slippage_pct)
					commission = (abs(open_position.size) * exit_price) * params.commission_pct
					pnl = (open_position.entry_price - exit_price) * abs(open_position.size) - commission
					equity += pnl
					tr_minutes = int((idx - open_position.entry_time).total_seconds() // 60)
					exit_reason = "stop" if stop_hit and not target_hit else ("target" if target_hit and not stop_hit else "stop_target_same_candle")
					open_position.exit_time = idx
					open_position.exit_price = exit_price
					open_position.pnl_usd = pnl
					open_position.pnl_pct = pnl / max(1e-9, initial_capital)
					open_position.holding_minutes = tr_minutes
					open_position.exit_reason = exit_reason
					trades.append(open_position)
					open_position = None

		# Record equity
		equity_curve.append(equity)
		curve_index.append(idx)

		# Determine entry signals on closed candle (we enter next candle open)
		if open_position is None:
			# Primary TF checks
			macd_ok = row.get("macd_hist", 0.0) > 0
			ema_ok = row.get("close", np.nan) > row.get("ema20", np.nan)
			vol_ok = True
			if "vol_ma" in row and row.get("vol_ma", np.nan) > 0 and "volume" in row:
				vol_ok = (row["volume"] >= params.vol_mult * row["vol_ma"]) if not math.isnan(row["vol_ma"]) else True

			long_primary = vwap_flip_long_condition(df5, idx) and macd_ok and ema_ok and vol_ok
			short_primary = vwap_flip_short_condition(df5, idx) and (row.get("macd_hist", 0.0) < 0) and (row.get("close", np.nan) < row.get("ema20", np.nan)) and vol_ok

			if long_primary or (params.allow_shorts and short_primary):
				direction = "long" if long_primary else "short"
				trend_15_ok, trend_1h_ok, disagree = higher_tf_alignment(df15, df1h, idx, direction)

				# Compute stop using VWAP +/- ATR
				atr_now = float(row.get("atr", np.nan))
				vwap_now = float(row.get("vwap", np.nan))
				if direction == "long":
					stop = vwap_now - params.atr_mult_for_stop * atr_now
				else:
					stop = vwap_now + params.atr_mult_for_stop * atr_now
				if not (stop == stop) or stop <= 0:
					continue

				# Entry at next candle open with slippage and commission
				if i + 1 >= len(df5.index):
					continue
				next_open_ts = df5.index[i + 1]
				next_open = float(df5.iloc[i + 1]["open"]) if direction == "long" else float(df5.iloc[i + 1]["open"])
				entry_price = next_open * (1.0 + (params.slippage_pct if direction == "long" else -params.slippage_pct))
				# Position sizing
				per_risk = abs(entry_price - stop)
				if per_risk <= 0:
					continue
				size = (equity * params.risk_per_trade) / per_risk
				if direction == "short":
					size = -size

				# Commission applied on entry and exit; subtract entry commission from equity right away
				entry_commission = abs(size) * entry_price * params.commission_pct
				equity -= entry_commission

				# Target
				if direction == "long":
					risk_per_unit = entry_price - stop
					target = entry_price + params.target_mult * risk_per_unit
				else:
					risk_per_unit = stop - entry_price
					target = entry_price - params.target_mult * risk_per_unit

				primary_ok = (long_primary if direction == "long" else short_primary)
				vol_mult_hit = vol_ok and (row.get("volume", np.nan) >= 1.5 * row.get("vol_ma", np.nan)) if ("vol_ma" in row and not math.isnan(row.get("vol_ma", np.nan))) else False
				confidence = compute_confidence(
					primary_ok=primary_ok,
					vol_multiplier_hit=bool(vol_mult_hit),
					macd_hist_value=float(row.get("macd_hist", 0.0)),
					macd_hist_top_quartile_threshold=macd_top_quartile,
					trend_15m_ok=trend_15_ok,
					trend_1h_ok=trend_1h_ok,
					higher_tf_disagree=disagree,
					mc_median_return_leq_zero=False,  # may be adjusted later using MC results
				)

				reason_parts = []
				reason_parts.append(f"VWAP flip on 5m={primary_ok}")
				reason_parts.append(f"vol={'NA' if 'vol_ma' not in row else (round(row.get('volume', 0.0) / max(1e-9, row.get('vol_ma', 1.0)), 2))}×")
				reason_parts.append(f"MACD_hist={round(float(row.get('macd_hist', 0.0)), 6)}")
				reason_parts.append(f"15m trend={'positive' if trend_15_ok else 'neutral' if not trend_15_ok else 'negative'}")
				reason_parts.append(f"1h trend={'positive' if trend_1h_ok else 'neutral' if not trend_1h_ok else 'negative'}")
				if disagree:
					reason_parts.append("higher TFs disagree")
				reason_parts.append("MC: not integrated in backtest confidence (runtime realtime may adjust)")
				primary_risk_drivers = []
				if get_instrument_type(instrument) == "EQUITY":
					primary_risk_drivers.append("SPX session volatility")
				if not vol_ok:
					primary_risk_drivers.append("low volume")
				recommendation = "PAPER only — low sample; do not scale"
				reason_text = "; ".join(reason_parts + [f"Risks: {', '.join(primary_risk_drivers) if primary_risk_drivers else 'standard'}", recommendation])

				open_position = TradeRecord(
					entry_time=next_open_ts,
					side=direction,
					entry_price=entry_price,
					size=float(size),
					stop=float(stop),
					target=float(target),
					exit_time=None,
					exit_price=None,
					pnl_usd=None,
					pnl_pct=None,
					holding_minutes=None,
					exit_reason=None,
					reason_text=reason_text,
				)

	# Close any open position at last close price (for stats completeness)
	if open_position is not None:
		last_row = df5.iloc[-1]
		exit_price = float(last_row["close"]) * (1.0 - params.slippage_pct if open_position.side == "long" else 1.0 + params.slippage_pct)
		commission = (abs(open_position.size) * exit_price) * params.commission_pct
		if open_position.side == "long":
			pnl = (exit_price - open_position.entry_price) * open_position.size - commission
		else:
			pnl = (open_position.entry_price - exit_price) * abs(open_position.size) - commission
		equity += pnl
		open_position.exit_time = df5.index[-1]
		open_position.exit_price = exit_price
		open_position.pnl_usd = pnl
		open_position.pnl_pct = pnl / max(1e-9, initial_capital)
		open_position.holding_minutes = int((df5.index[-1] - open_position.entry_time).total_seconds() // 60)
		open_position.exit_reason = "eod_close"
		trades.append(open_position)
		open_position = None
		equity_curve.append(equity)
		curve_index.append(df5.index[-1])

	equity_series = pd.Series(equity_curve, index=pd.to_datetime(curve_index, utc=True) if len(curve_index) else None)
	if equity_series.empty:
		drawdown = equity_series
	else:
		rolling_max = equity_series.cummax()
		dd = (equity_series - rolling_max) / rolling_max.replace(0.0, np.nan)
		drawdown = dd

	stats = summarize_trades(trades, initial_capital)
	return BacktestResult(trades=trades, equity_curve=equity_series, drawdown_curve=drawdown, stats=stats)


def summarize_trades(trades: List[TradeRecord], initial_capital: float) -> Dict[str, float]:
	if not trades:
		return {
			"num_trades": 0,
			"win_rate": 0.0,
			"avg_pnl": 0.0,
			"expectancy": 0.0,
			"total_return_pct": 0.0,
			"max_drawdown_pct": 0.0,
		}
	wins = [t for t in trades if (t.pnl_usd or 0.0) > 0]
	losses = [t for t in trades if (t.pnl_usd or 0.0) <= 0]
	total_pnl = sum([(t.pnl_usd or 0.0) for t in trades])
	win_rate = len(wins) / max(1, len(trades))
	avg_pnl = total_pnl / max(1, len(trades))
	expectancy = avg_pnl  # per trade USD expectancy
	total_return_pct = total_pnl / max(1e-9, initial_capital)
	# Approximate max drawdown from equity path reconstructed from pnl
	equity = initial_capital
	path = []
	for t in trades:
		equity += (t.pnl_usd or 0.0)
		path.append(equity)
	if not path:
		max_dd = 0.0
	else:
		arr = np.array(path)
		peak = np.maximum.accumulate(arr)
		dd = (arr - peak) / np.where(peak == 0.0, 1.0, peak)
		max_dd = float(dd.min())
	return {
		"num_trades": float(len(trades)),
		"win_rate": float(win_rate),
		"avg_pnl": float(avg_pnl),
		"expectancy": float(expectancy),
		"total_return_pct": float(total_return_pct),
		"max_drawdown_pct": float(max_dd),
	}


# ------------------------ PLOTTING ------------------------

def plot_equity_and_drawdown(outdir: str, instrument: str, bt: BacktestResult) -> None:
	if bt.equity_curve.empty:
		return
	plt.figure(figsize=(10, 4))
	bt.equity_curve.plot()
	plt.title(f"Equity Curve - {instrument}")
	plt.xlabel("Time")
	plt.ylabel("Equity (USD)")
	plt.tight_layout()
	plt.savefig(os.path.join(outdir, f"bt_equity_curve_{instrument}.png"))
	plt.close()

	plt.figure(figsize=(10, 3))
	bt.drawdown_curve.plot(color="red")
	plt.title(f"Drawdown - {instrument}")
	plt.xlabel("Time")
	plt.ylabel("Drawdown")
	plt.tight_layout()
	plt.savefig(os.path.join(outdir, f"bt_drawdown_{instrument}.png"))
	plt.close()


def plot_mc_hist(outdir: str, instrument: str, slippage: float, returns: List[float]) -> None:
	plt.figure(figsize=(8, 4))
	plt.hist(returns, bins=40, color="#4477aa", alpha=0.8)
	plt.title(f"MC Returns Distribution - {instrument} - slippage={slippage}")
	plt.xlabel("Total Return (%)")
	plt.ylabel("Frequency")
	plt.tight_layout()
	plt.savefig(os.path.join(outdir, f"mc_returns_hist_{instrument}_{slippage}.png"))
	plt.close()


# ------------------------ MONTE CARLO ------------------------

def split_by_session_day(df5: pd.DataFrame, instrument: str) -> Dict[str, pd.DataFrame]:
	if df5.empty:
		return {}
	df5 = df5.copy()
	if df5.index.tz is None:
		df5.index = df5.index.tz_localize("UTC")
	df5["session_id"] = [session_id_for_row(ts, instrument) for ts in df5.index]
	groups: Dict[str, pd.DataFrame] = {}
	for sid, g in df5.groupby("session_id"):
		groups[sid] = g.drop(columns=["session_id"], errors="ignore")
	return groups


def bootstrap_mc(
	instrument: str,
	base_df5: pd.DataFrame,
	df15: Optional[pd.DataFrame],
	df1h: Optional[pd.DataFrame],
	params: StrategyParams,
	initial_capital: float,
	slippage_scenarios: List[float],
	trials: int,
	outdir: str,
	quiet: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
	"""Return nested dict: {slippage: {param_key: stats}} and write hist plots."""
	day_blocks = split_by_session_day(base_df5, instrument)
	day_keys = list(day_blocks.keys())
	if not day_keys:
		return {}

	results: Dict[str, Dict[str, Dict[str, float]]] = {}
	for slip in slippage_scenarios:
		all_returns_this_slip: Dict[str, List[float]] = {}
		results[str(slip)] = {}
		for atr_mult in PARAM_SWEEP_ATR_MULT:
			for vol_mult in PARAM_SWEEP_VOL_MULT:
				for risk in PARAM_SWEEP_RISK:
					key = f"atr={atr_mult}_vol={vol_mult}_risk={risk}"
					all_ret_pct: List[float] = []
					all_num_trades: List[float] = []
					all_mdd: List[float] = []
					for _ in tqdm(range(trials), disable=quiet, desc=f"MC {instrument} slip={slip} {key}"):
						# Sample N day-blocks with replacement
						n = len(day_keys)
						picked = [day_blocks[random.choice(day_keys)] for _ in range(n)]
						trial_df5 = pd.concat(picked, axis=0)
						trial_df5 = add_session_vwap(trial_df5, instrument)
						trial_df5 = add_indicators(trial_df5, StrategyParams(
							atr_len=params.atr_len,
							ema_len=params.ema_len,
							vol_ma_len=params.vol_ma_len,
							vol_mult=vol_mult,
							atr_mult_for_stop=atr_mult,
							target_mult=params.target_mult,
							risk_per_trade=risk,
							commission_pct=params.commission_pct,
							slippage_pct=slip,
							allow_shorts=params.allow_shorts,
						))
						bt = simulate_trades(instrument, trial_df5, df15, df1h, StrategyParams(
							atr_len=params.atr_len,
							ema_len=params.ema_len,
							vol_ma_len=params.vol_ma_len,
							vol_mult=vol_mult,
							atr_mult_for_stop=atr_mult,
							target_mult=params.target_mult,
							risk_per_trade=risk,
							commission_pct=params.commission_pct,
							slippage_pct=slip,
							allow_shorts=params.allow_shorts,
						), initial_capital, quiet=True)
						stats = bt.stats
						all_ret_pct.append(stats.get("total_return_pct", 0.0) * 100.0)
						all_num_trades.append(stats.get("num_trades", 0.0))
						all_mdd.append(stats.get("max_drawdown_pct", 0.0) * 100.0)
					# Summaries
					returns_np = np.array(all_ret_pct) if all_ret_pct else np.array([0.0])
					mdd_np = np.array(all_mdd) if all_mdd else np.array([0.0])
					tr_np = np.array(all_num_trades) if all_num_trades else np.array([0.0])
					results[str(slip)][key] = {
						"pctile_5": float(np.percentile(returns_np, 5)),
						"pctile_25": float(np.percentile(returns_np, 25)),
						"pctile_50": float(np.percentile(returns_np, 50)),
						"pctile_75": float(np.percentile(returns_np, 75)),
						"pctile_95": float(np.percentile(returns_np, 95)),
						"median_max_drawdown_pct": float(np.percentile(mdd_np, 50)),
						"avg_trades": float(np.mean(tr_np)),
						"pct_profitable_trials": float((returns_np > 0).mean() * 100.0),
					}
					all_returns_this_slip[key] = all_ret_pct
		# Plot histogram for base params key if present
		# Choose base key atr=1.0_vol=1.25_risk=0.01
		base_key = "atr=1.0_vol=1.25_risk=0.01"
		if base_key in all_returns_this_slip:
			plot_mc_hist(outdir, instrument, slip, all_returns_this_slip[base_key])
	return results


# ------------------------ FILE OUTPUT ------------------------

def write_trade_logs(outdir: str, instrument: str, bt: BacktestResult) -> None:
	if not bt.trades:
		# Still create empty files with headers
		columns = [
			"entry_time","side","entry_price","size","stop","target","exit_time","exit_price","pnl_usd","pnl_pct","holding_minutes","exit_reason","reason_text"
		]
		empty_df = pd.DataFrame(columns=columns)
		empty_df.to_csv(os.path.join(outdir, f"signals_backtest_{instrument}.csv"), index=False)
		empty_df.to_csv(os.path.join(outdir, f"trades_backtest_{instrument}.csv"), index=False)
		return
	rows = []
	for t in bt.trades:
		rows.append({
			"entry_time": t.entry_time.isoformat(),
			"side": t.side,
			"entry_price": t.entry_price,
			"size": t.size,
			"stop": t.stop,
			"target": t.target,
			"exit_time": t.exit_time.isoformat() if t.exit_time else "",
			"exit_price": t.exit_price if t.exit_price is not None else "",
			"pnl_usd": t.pnl_usd if t.pnl_usd is not None else "",
			"pnl_pct": t.pnl_pct if t.pnl_pct is not None else "",
			"holding_minutes": t.holding_minutes if t.holding_minutes is not None else "",
			"exit_reason": t.exit_reason if t.exit_reason is not None else "",
			"reason_text": t.reason_text,
		})
	df = pd.DataFrame(rows)
	df.to_csv(os.path.join(outdir, f"signals_backtest_{instrument}.csv"), index=False)
	df.to_csv(os.path.join(outdir, f"trades_backtest_{instrument}.csv"), index=False)


def write_mc_summary(outdir: str, instrument: str, mc_results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
	path = os.path.join(outdir, f"mc_summary_{instrument}.json")
	with open(path, "w") as f:
		json.dump(mc_results, f, indent=2)


# ------------------------ REALTIME ------------------------

class RealtimeEngine:
	"""Realtime signal engine for BTC via Binance WS and local rolling CSVs for others."""

	def __init__(self,
				outdir: str,
				params: StrategyParams,
				csv_paths: Dict[str, Optional[str]],
				mc_summaries: Dict[str, Dict],
				quiet: bool = False) -> None:
		self.outdir = outdir
		self.params = params
		self.csv_paths = csv_paths
		self.mc_summaries = mc_summaries
		self.quiet = quiet

		self.lock = threading.Lock()
		self.data: Dict[str, Dict[str, pd.DataFrame]] = {
			"BTC-USD": {tf: pd.DataFrame() for tf in DEFAULT_TIMEFRAMES},
			"XAU-USD": {tf: pd.DataFrame() for tf in DEFAULT_TIMEFRAMES},
			"EUR-USD": {tf: pd.DataFrame() for tf in DEFAULT_TIMEFRAMES},
			"SPX": {tf: pd.DataFrame() for tf in DEFAULT_TIMEFRAMES},
		}

		self.realtime_signals_path = os.path.join(outdir, "realtime_signals.csv")
		self.dashboard_path = os.path.join(outdir, "realtime_dashboard.json")
		self._init_output_files()

	def _init_output_files(self) -> None:
		if not os.path.exists(self.realtime_signals_path):
			with open(self.realtime_signals_path, "w") as f:
				f.write("timestamp,instrument,timeframe,side,price,size,stop,target,confidence_pct,reason_text\n")
		# write initial dashboard
		with open(self.dashboard_path, "w") as f:
			json.dump({"status": "initializing"}, f)

	def start(self) -> None:
		threads: List[threading.Thread] = []
		# BTC via Binance WS if possible
		if WebSocketApp is not None:
			ws_thread = threading.Thread(target=self._run_binance_ws, daemon=True)
			threads.append(ws_thread)
			ws_thread.start()
		else:
			self._log("websocket-client not installed; BTC realtime disabled")
		# Others via polling local rolling CSVs
		for inst in ["XAU-USD", "EUR-USD", "SPX"]:
			path = self.csv_paths.get(inst)
			if path:
				thr = threading.Thread(target=self._poll_local_csv, args=(inst, path), daemon=True)
				threads.append(thr)
				thr.start()
			else:
				self._log(f"No realtime feed for {inst}; using static data only if provided")
		# Main loop: every minute check 5m closures and emit signals
		try:
			while True:
				self._check_and_emit(PRIMARY_TF)
				time.sleep(10)
		except KeyboardInterrupt:
			self._log("Realtime stopped by user")

	def _run_binance_ws(self) -> None:  # pragma: no cover
		def on_message(ws, message):
			try:
				data = json.loads(message)
				stream = data.get("stream", "")
				k = data.get("data", {}).get("k", {})
				if not k:
					return
				tf = k.get("i", "")
				# Map Binance tf to our set
				if tf not in ["1m", "5m", "15m", "1h"]:
					return
				bar = {
					"datetime": pd.to_datetime(int(k.get("t", 0)), unit="ms", utc=True),
					"open": float(k.get("o", 0.0)),
					"high": float(k.get("h", 0.0)),
					"low": float(k.get("l", 0.0)),
					"close": float(k.get("c", 0.0)),
					"volume": float(k.get("v", 0.0)),
					"closed": bool(k.get("x", False)),
				}
				self._append_bar("BTC-USD", tf, bar)
			except Exception as e:
				self._log(f"WS message error: {e}")

		def on_error(ws, error):
			self._log(f"WS error: {error}")

		def on_close(ws, close_status_code, close_msg):
			self._log("WS closed")

		def on_open(ws):
			self._log("WS opened")

		ws = WebSocketApp(BINANCE_WS_URL, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
		ws.run_forever()

	def _append_bar(self, instrument: str, tf: str, bar: Dict) -> None:
		with self.lock:
			df = self.data[instrument][tf]
			new_row = pd.DataFrame([{
				"datetime": bar["datetime"],
				"open": bar["open"],
				"high": bar["high"],
				"low": bar["low"],
				"close": bar["close"],
				"volume": bar["volume"],
			}]).set_index("datetime")
			if df is None or df.empty:
				df = new_row
			else:
				df = pd.concat([df, new_row])
			# On closed candles, compute indicators
			self.data[instrument][tf] = add_indicators(add_session_vwap(df, instrument), self.params)

	def _poll_local_csv(self, instrument: str, path: str) -> None:
		last_mtime = 0.0
		while True:
			try:
				mtime = os.path.getmtime(path)
				if mtime != last_mtime:
					df = read_history_csv(path)
					if df is not None and not df.empty:
						# Assume CSV is for PRIMARY_TF; resample higher TFs
						with self.lock:
							df5 = add_indicators(add_session_vwap(df, instrument), self.params)
							df15 = add_indicators(add_session_vwap(resample_ohlcv(df, "15m"), instrument), self.params)
							df1h = add_indicators(add_session_vwap(resample_ohlcv(df, "1h"), instrument), self.params)
							self.data[instrument]["5m"] = df5
							self.data[instrument]["15m"] = df15
							self.data[instrument]["1h"] = df1h
					last_mtime = mtime
			except Exception as e:
				self._log(f"poll {instrument} error: {e}")
			time.sleep(5)

	def _check_and_emit(self, primary_tf: str) -> None:
		with self.lock:
			for instrument in INSTRUMENTS:
				df5 = self.data[instrument].get("5m")
				df15 = self.data[instrument].get("15m")
				df1h = self.data[instrument].get("1h")
				if df5 is None or df5.empty:
					continue
				# Only act on last fully closed candle
				if len(df5.index) < 2:
					continue
				idx = df5.index[-2]
				row = df5.loc[idx]
				macd_ok = row.get("macd_hist", 0.0) > 0
				ema_ok = row.get("close", np.nan) > row.get("ema20", np.nan)
				vol_ok = True
				if "vol_ma" in row and row.get("vol_ma", np.nan) > 0 and "volume" in row:
					vol_ok = (row["volume"] >= self.params.vol_mult * row["vol_ma"]) if not math.isnan(row["vol_ma"]) else True
				long_primary = vwap_flip_long_condition(df5, idx) and macd_ok and ema_ok and vol_ok
				short_primary = vwap_flip_short_condition(df5, idx) and (row.get("macd_hist", 0.0) < 0) and (row.get("close", np.nan) < row.get("ema20", np.nan)) and vol_ok
				if not long_primary and not (self.params.allow_shorts and short_primary):
					continue
				direction = "long" if long_primary else "short"
				trend_15_ok, trend_1h_ok, disagree = higher_tf_alignment(df15, df1h, idx, direction)
				# Stop at current vwap +/- atr
				atr_now = float(row.get("atr", np.nan))
				vwap_now = float(row.get("vwap", np.nan))
				if direction == "long":
					stop = vwap_now - self.params.atr_mult_for_stop * atr_now
					target = row["close"] + self.params.target_mult * (row["close"] - stop)
				else:
					stop = vwap_now + self.params.atr_mult_for_stop * atr_now
					target = row["close"] - self.params.target_mult * (stop - row["close"]) 
				per_risk = abs(row["close"] - stop)
				if per_risk <= 0:
					continue
				size = (1000.0 * self.params.risk_per_trade) / per_risk
				if direction == "short":
					size = -size
				macd_hist_abs = df5["macd_hist"].abs().dropna()
				macd_top_quartile = float(np.percentile(macd_hist_abs, 75)) if not macd_hist_abs.empty else 0.0
				# MC penalty if median return <= 0 for avg slippage scenario
				mc_penalty = False
				mc_summary = self.mc_summaries.get(instrument)
				if mc_summary is not None:
					avg_slip = DEFAULT_SLIPPAGE_SCENARIOS[1]
					base_key = "atr=1.0_vol=1.25_risk=0.01"
					try:
						median = mc_summary[str(avg_slip)][base_key]["pctile_50"]
						mc_penalty = median <= 0
					except Exception:
						mc_penalty = False
				confidence = compute_confidence(
					primary_ok=True,
					vol_multiplier_hit=bool("vol_ma" in row and row.get("vol_ma", np.nan) > 0 and row.get("volume", 0.0) >= 1.5 * row.get("vol_ma", np.nan)),
					macd_hist_value=float(row.get("macd_hist", 0.0)),
					macd_hist_top_quartile_threshold=macd_top_quartile,
					trend_15m_ok=trend_15_ok,
					trend_1h_ok=trend_1h_ok,
					higher_tf_disagree=disagree,
					mc_median_return_leq_zero=mc_penalty,
				)
				reason_parts = []
				reason_parts.append("VWAP flip on 5m")
				reason_parts.append(f"vol={'NA' if 'vol_ma' not in row else (round(row.get('volume', 0.0) / max(1e-9, row.get('vol_ma', 1.0)), 2))}×")
				reason_parts.append(f"MACD_hist={round(float(row.get('macd_hist', 0.0)), 6)}")
				reason_parts.append(f"15m trend={'positive' if trend_15_ok else 'neutral'}")
				reason_parts.append(f"1h trend={'positive' if trend_1h_ok else 'neutral'}")
				if disagree:
					reason_parts.append("higher TFs disagree")
				if mc_penalty:
					reason_parts.append("MC: median <= 0 at avg slippage")
				recommendation = "PAPER only — low sample; do not scale"
				reason_text = "; ".join(reason_parts + [f"Risks: {'standard'}", recommendation])
				# Emit
				self._append_realtime_signal(idx, instrument, PRIMARY_TF, direction, float(row["close"]), float(size), float(stop), float(target), confidence, reason_text)

		# Update dashboard
		self._write_dashboard_snapshot()

	def _append_realtime_signal(self, ts: pd.Timestamp, instrument: str, tf: str, side: str, price: float, size: float, stop: float, target: float, confidence: int, reason_text: str) -> None:
		line = f"{ts.isoformat()},{instrument},{tf},{side},{price},{size},{stop},{target},{confidence},{reason_text}\n"
		with open(self.realtime_signals_path, "a") as f:
			f.write(line)
		self._log(f"Signal {instrument} {tf} {side} price={price} conf={confidence}%")

	def _write_dashboard_snapshot(self) -> None:
		snapshot = {"timestamp": datetime.now(timezone.utc).isoformat(), "tf": PRIMARY_TF, "instruments": {}}
		with self.lock:
			for inst in INSTRUMENTS:
				df5 = self.data[inst].get("5m")
				if df5 is None or df5.empty:
					continue
				row = df5.iloc[-1]
				snapshot["instruments"][inst] = {
					"last_close": float(row.get("close", np.nan)),
					"ema20": float(row.get("ema20", np.nan)),
					"vwap": float(row.get("vwap", np.nan)),
					"macd_hist": float(row.get("macd_hist", np.nan)),
				}
		with open(self.dashboard_path, "w") as f:
			json.dump(snapshot, f, indent=2)

	def _log(self, msg: str) -> None:
		if not self.quiet:
			print(msg)


# ------------------------ HIGH-LEVEL RUNNERS ------------------------

def prepare_timeframes(df5: pd.DataFrame, instrument: str, params: StrategyParams) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Given 5m data, return enriched 5m, 15m, 1h frames with indicators and session VWAP."""
	df5e = add_indicators(add_session_vwap(df5, instrument), params)
	df15 = add_indicators(add_session_vwap(resample_ohlcv(df5, "15m"), instrument), params)
	df1h = add_indicators(add_session_vwap(resample_ohlcv(df5, "1h"), instrument), params)
	return df5e, df15, df1h


def run_backtest_for_instrument(instrument: str, df5: pd.DataFrame, params: StrategyParams, initial_capital: float, outdir: str, trials: int, quiet: bool = False) -> Tuple[BacktestResult, Dict[str, Dict[str, Dict[str, float]]]]:
	df5e, df15, df1h = prepare_timeframes(df5, instrument, params)
	bt = simulate_trades(instrument, df5e, df15, df1h, params, initial_capital, quiet=quiet)
	plot_equity_and_drawdown(outdir, instrument, bt)
	write_trade_logs(outdir, instrument, bt)
	mc_results = bootstrap_mc(instrument, df5e, df15, df1h, params, initial_capital, DEFAULT_SLIPPAGE_SCENARIOS, trials=trials, outdir=outdir, quiet=quiet)
	write_mc_summary(outdir, instrument, mc_results)
	return bt, mc_results


def build_combined_actionability(outdir: str, last_confidences: Dict[str, int], mc_summaries: Dict[str, Dict]) -> None:
	lines = []
	lines.append("Instrument,Actionability(0-10),Recommendation")
	for inst in INSTRUMENTS:
		conf = last_confidences.get(inst, 30)
		# Base score from confidence
		score = max(0, min(10, int(round(conf / 10))))
		rec = "PAPER only — low sample; do not scale"
		# If MC shows robust results, slightly upgrade recommendation
		mc = mc_summaries.get(inst)
		if mc:
			try:
				avg_slip = str(DEFAULT_SLIPPAGE_SCENARIOS[1])
				base_key = "atr=1.0_vol=1.25_risk=0.01"
				median = mc[avg_slip][base_key]["pctile_50"]
				pct_profitable = mc[avg_slip][base_key]["pct_profitable_trials"]
				avg_trades = mc[avg_slip][base_key]["avg_trades"]
				if median > 0 and pct_profitable >= 70 and avg_trades >= DEFAULT_MIN_TRADES_CONFIDENCE:
					score = min(10, score + 2)
					rec = "Small live OK — test with 0.5× size"
			except Exception:
				pass
		lines.append(f"{inst},{score},{rec}")
	with open(os.path.join(outdir, "combined_actionability.txt"), "w") as f:
		f.write("\n".join(lines))


# ------------------------ MAIN ------------------------

def main() -> None:
	args = parse_args()
	timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
	params = StrategyParams(
		atr_len=DEFAULT_ATR_LEN,
		ema_len=DEFAULT_EMA_LEN,
		vol_ma_len=DEFAULT_VOL_MA_LEN,
		vol_mult=DEFAULT_VOL_MULT,
		atr_mult_for_stop=DEFAULT_ATR_MULT_FOR_STOP,
		target_mult=DEFAULT_TARGET_MULT,
		risk_per_trade=args.risk,
		commission_pct=DEFAULT_COMMISSION_PCT,
		slippage_pct=args.slippage,
		allow_shorts=args.allow_shorts,
	)
	os.makedirs(args.outdir, exist_ok=True)

	print(RISK_WARNING_HEADER)

	csv_map = {
		"BTC-USD": args.csv_btc,
		"XAU-USD": args.csv_xau,
		"EUR-USD": args.csv_fx,
		"SPX": args.csv_spx,
	}

	loaded: Dict[str, Optional[pd.DataFrame]] = {}
	for inst, path in csv_map.items():
		try:
			loaded[inst] = read_history_csv(path, timezone_name=args.timezone_override)
		except Exception as e:
			print(f"Error reading {inst} CSV {path}: {e}")
			loaded[inst] = None

	if args.mode in ("backtest", "both"):
		last_conf: Dict[str, int] = {}
		mc_summaries: Dict[str, Dict] = {}
		for inst in INSTRUMENTS:
			df5 = loaded.get(inst)
			if df5 is None or df5.empty:
				print(f"WARNING: No or empty CSV for {inst}. Skipping backtest+MC; realtime can still run if feed present.")
				continue
			# Enforce 7 contiguous days check (warn only)
			days_span = (df5.index.max() - df5.index.min()).days
			if days_span < 6:
				print(f"WARNING: {inst} CSV < 7 days. Statistical power low; MC confidence reduced.")

			bt, mc = run_backtest_for_instrument(inst, df5, params, args.initial_capital, args.outdir, trials=args.mc, quiet=args.quiet)
			mc_summaries[inst] = mc
			# Compute a notional last confidence from final candle
			if not df5.empty:
				# Use end-of-backtest candle for a dry confidence score (no MC penalty here)
				df5e, df15, df1h = prepare_timeframes(df5, inst, params)
				idx = df5e.index[-1]
				row = df5e.iloc[-1]
				macd_hist_abs = df5e["macd_hist"].abs().dropna()
				macd_top_quartile = float(np.percentile(macd_hist_abs, 75)) if not macd_hist_abs.empty else 0.0
				trend_15_ok, trend_1h_ok, disagree = higher_tf_alignment(df15, df1h, idx, "long" if row.get("close", 0.0) > row.get("ema20", 0.0) else "short")
				primary_ok = (vwap_flip_long_condition(df5e, idx) and row.get("macd_hist", 0.0) > 0 and row.get("close", 0.0) > row.get("ema20", 0.0))
				last_conf[inst] = compute_confidence(primary_ok, bool(row.get("volume", 0.0) >= 1.5 * row.get("vol_ma", 1.0) if row.get("vol_ma", np.nan) == row.get("vol_ma", np.nan) else False), float(row.get("macd_hist", 0.0)), macd_top_quartile, trend_15_ok, trend_1h_ok, disagree, False)
		build_combined_actionability(args.outdir, last_conf, mc_summaries)

	if args.mode in ("realtime", "both"):
		# Load MC summaries if exist to feed realtime confidence penalty
		mc_summaries: Dict[str, Dict] = {}
		for inst in INSTRUMENTS:
			path = os.path.join(args.outdir, f"mc_summary_{inst}.json")
			if os.path.exists(path):
				try:
					with open(path, "r") as f:
						mc_summaries[inst] = json.load(f)
				except Exception:
					pass
		engine = RealtimeEngine(
			outdir=args.outdir,
			params=params,
			csv_paths={
				"BTC-USD": args.csv_btc if os.path.exists(args.csv_btc) else None,
				"XAU-USD": args.csv_xau if os.path.exists(args.csv_xau) else None,
				"EUR-USD": args.csv_fx if os.path.exists(args.csv_fx) else None,
				"SPX": args.csv_spx if os.path.exists(args.csv_spx) else None,
			},
			mc_summaries=mc_summaries,
			quiet=args.quiet,
		)
		engine.start()


if __name__ == "__main__":
	main()