# main.py
import io,inspect
import os
import json,itertools
import traceback
import logging
from typing import Optional, Tuple, Dict, Any
import requests
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from functools import lru_cache
# from googlesearch import search
# from newspaper import Article
# import nltk
import schedule
from datetime import datetime, timezone
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math
import time
# nltk.download("punkt")
# nltk.download("stopwords")

# ---------- config ----------
load_dotenv()
API_KEY = os.getenv("MY_SECRET_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", API_KEY)
AV_BASE_URL = "https://www.alphavantage.co/query"

# Key rotation, Loading all keys from env
keys_raw = os.getenv("ALPHA_VANTAGE_KEYS", "")
ALPHA_KEYS = [k.strip() for k in keys_raw.split(",") if k.strip()]

# Create a round-robin cycle iterator
key_cycle = itertools.cycle(ALPHA_KEYS)

def get_next_av_key():
    """Return the next API key in round-robin fashion"""
    return next(key_cycle)
    
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-predict")

# FastAPI
app = FastAPI(
    title="📈 Stock Price Prediction API - Debuggable",
    description="Predict stock prices using a trained BiLSTM model (inference-only).",
    version="2.1.0",
)
#  Allow your Vercel frontend
origins = [
    "https://market-trend-analysis.vercel.app", 
    "http://localhost:8080",                     # for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- model loader (lazy, cached) ----------
@lru_cache()
def get_model():
    """
    Load Keras model + scalers once (cached).
    Files expected: mod/model.keras, mod/scaler_X.pkl, mod/scaler_y.pkl
    """
    try:
        import tensorflow as tf  # imported here to avoid importing TF at module import time for quick linting
        from tensorflow import keras
        model = keras.models.load_model("mod/model.keras", compile=False)
        scaler_X = joblib.load("mod/scaler_X.pkl")
        scaler_y = joblib.load("mod/scaler_y.pkl")
        logger.info("Model and scalers loaded from disk")
        return model, scaler_X, scaler_y
    except Exception as e:
        logger.exception("Failed to load model or scalers: %s", e)
        raise


# ---------- utilities ----------
def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If df has MultiIndex columns (e.g. from yfinance returning ('Close','AAPL')), flatten them."""
    if isinstance(df.columns, pd.MultiIndex):
        # prefer the first level (e.g. 'Close'); if duplicates remain, append second level
        lvl0 = df.columns.get_level_values(0).astype(str)
        lvl1 = df.columns.get_level_values(1).astype(str)
        new_cols = []
        seen = {}
        for a, b in zip(lvl0, lvl1):
            name = a
            # if duplicate, append ticker/supplement
            if name in seen:
                name = f"{a}_{b}" if b and b != "None" else a
            seen[name] = True
            new_cols.append(name)
        df.columns = new_cols
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names. Internal canonical is short names: 'open','high','low','close','volume'.
    Also create long-style aliases like '1. open' <-> 'open' when possible.
    """
    df = df.copy()
    # flatten MultiIndex early
    df = _flatten_multiindex_columns(df)

    # map common variants to canonical long style '1. open' etc.
    orig = list(df.columns)
    cols = [str(c) for c in orig]
    mapping = {}
    for c_raw, c in zip(orig, cols):
        k = __import__("re").sub(r"[^0-9a-z]", "", c.lower())
        if "1open" in k or k == "open":
            mapping[c_raw] = "1. open"
        elif "2high" in k or k == "high":
            mapping[c_raw] = "2. high"
        elif "3low" in k or k == "low":
            mapping[c_raw] = "3. low"
        elif "4close" in k or k == "close":
            mapping[c_raw] = "4. close"
        elif "5volume" in k or k == "volume":
            mapping[c_raw] = "5. volume"
        elif "adj" in k and "close" in k:
            mapping[c_raw] = "adjclose"
    if mapping:
        df = df.rename(columns=mapping)

    # create short aliases if long names exist (and vice-versa)
    alias_pairs = [
        ("1. open", "open"),
        ("2. high", "high"),
        ("3. low", "low"),
        ("4. close", "close"),
        ("5. volume", "volume"),
    ]
    for long_name, short_name in alias_pairs:
        if long_name in df.columns and short_name not in df.columns:
            col = df[long_name]
            # if column is a DataFrame (rare), pick first column
            if isinstance(col, pd.DataFrame):
                df[short_name] = col.iloc[:, 0]
            else:
                df[short_name] = col
        elif short_name in df.columns and long_name not in df.columns:
            col = df[short_name]
            if isinstance(col, pd.DataFrame):
                df[long_name] = col.iloc[:, 0]
            else:
                df[long_name] = col

    # remove exact duplicate column labels (keep the first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    # ensure adjclose exists as fallback
    if "adjclose" not in df.columns and "4. close" in df.columns:
        df["adjclose"] = df["4. close"]

    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))


def safe_read_csv_from_url(url: str, timeout: int = 10) -> pd.DataFrame:
    """
    Read CSV using requests to control timeouts and raise clear errors for AlphaVantage JSON/HTML responses.
    Returns a raw DataFrame (no normalization here).
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    text = resp.text.strip()

    # AlphaVantage sometimes returns JSON with {"Note": "..."} or {"Error Message": "..."}
    if text.startswith("{") or text.startswith("["):
        try:
            info = resp.json()
        except Exception:
            raise RuntimeError("AlphaVantage returned JSON but cannot parse it")
        msg = info.get("Note") or info.get("Error Message") or json.dumps(info)
        raise RuntimeError(f"AlphaVantage returned error: {msg}")

    # HTML error page
    if text.startswith("<"):
        raise RuntimeError("AlphaVantage returned HTML (likely rate limited or blocked).")

    # parse CSV
    df = pd.read_csv(io.StringIO(resp.text))
    return df


def _safe_scalar(val: Any):
    """
    Convert possible Series/ndarray-like cell to a scalar:
      - If val is scalar -> return as-is
      - If val is Series/ndarray -> return its first element (or np.nan if empty)
    """
    if isinstance(val, pd.Series):
        if val.empty:
            return np.nan
        return val.iat[0]
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val)
        if arr.size == 0:
            return np.nan
        return arr.ravel()[0]
    return val


import itertools
import inspect

def fetch_stock_data(symbol: str, outputsize: str = "compact", timeout: int = 10) -> Tuple[pd.DataFrame, Dict]:
    """
    Robust fetch:
     1) Try AlphaVantage using every key in ALPHA_KEYS (round-robin).
     2) If AV fails for all keys, robustly try yfinance with several signatures / retries.
     3) If yfinance fails, try direct Yahoo chart JSON with a browser-like User-Agent.
    Returns normalized df and charts dict (same structure as before).
    """
    e_av = None
    e_yf = None
    e_yahoo_json = None
    df = None
    charts = {}

    # helper to finalize df + chart building (keeps your previous logic)
    def _finalize_df_and_charts(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df_loc = df_in.copy()
        df_loc = _flatten_multiindex_columns(df_loc)
        df_loc.index = pd.to_datetime(df_loc.index, errors="coerce")
        df_loc = _normalize_columns(df_loc)
        df_loc = df_loc.sort_index()

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df_loc.columns]
        if missing:
            raise ValueError(f"Missing required column(s) {missing} after normalization. Available: {list(df_loc.columns)}")

        df_loc["ma20"] = df_loc["close"].rolling(window=20).mean()
        df_loc["ma50"] = df_loc["close"].rolling(window=50).mean()
        df_loc["rsi"] = calculate_rsi(df_loc["close"])

        if not pd.api.types.is_datetime64_any_dtype(df_loc.index):
            df_loc.index = pd.to_datetime(df_loc.index, errors="coerce")
            if df_loc.index.isnull().any():
                df_loc.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df_loc))

        df_monthly = df_loc.resample("M").last()
        price_data, volume_data, rsi_data = [], [], []
        for d, row in df_monthly.iterrows():
            close_val = _safe_scalar(row.get("close") if "close" in row.index else row.get("4. close"))
            ma20_val = _safe_scalar(row.get("ma20"))
            ma50_val = _safe_scalar(row.get("ma50"))
            vol_val = _safe_scalar(row.get("volume") if "volume" in row.index else row.get("5. volume"))
            rsi_val = _safe_scalar(row.get("rsi"))

            price_data.append({
                "date": d.strftime("%b %Y"),
                "close": round(float(close_val), 2) if pd.notna(close_val) else None,
                "ma20": round(float(ma20_val), 2) if pd.notna(ma20_val) else None,
                "ma50": round(float(ma50_val), 2) if pd.notna(ma50_val) else None,
            })

            try:
                vol_int = int(float(vol_val)) if pd.notna(vol_val) else None
            except Exception:
                vol_int = None
            volume_data.append({"date": d.strftime("%b %Y"), "volume": vol_int})

            rsi_data.append({"date": d.strftime("%b %Y"), "rsi": round(float(rsi_val), 2) if pd.notna(rsi_val) else None})

        return df_loc, {"price_data": price_data, "volume_data": volume_data, "rsi_data": rsi_data}

    # ---------- 1) Try AlphaVantage using all available keys ----------
    if ALPHA_KEYS:
        keys_tried = 0
        for _ in range(len(ALPHA_KEYS)):
            av_key = get_next_av_key()
            keys_tried += 1
            try:
                url = (
                    f"{AV_BASE_URL}?function=TIME_SERIES_DAILY&symbol={symbol}"
                    f"&outputsize={outputsize}&apikey={av_key}&datatype=csv"
                )
                logger.info("Fetching CSV from AlphaVantage with key=%s: %s", av_key, url)
                df_try = safe_read_csv_from_url(url, timeout=timeout)
                if df_try is None or getattr(df_try, "empty", True):
                    raise RuntimeError("AlphaVantage returned empty CSV")

                # index normalization
                if "timestamp" in df_try.columns:
                    df_try["timestamp"] = pd.to_datetime(df_try["timestamp"], errors="coerce")
                    df_try = df_try.set_index("timestamp")
                elif "date" in df_try.columns:
                    df_try["date"] = pd.to_datetime(df_try["date"], errors="coerce")
                    df_try = df_try.set_index("date")

                df_try = _normalize_columns(df_try)
                df_try = df_try.sort_index()
                if df_try.empty:
                    raise RuntimeError("AlphaVantage returned empty after normalization")

                df_final, charts = _finalize_df_and_charts(df_try)
                logger.info("AlphaVantage succeeded for %s using key %s (%d rows)", symbol, av_key, len(df_final))
                return df_final, charts

            except Exception as exc_av:
                e_av = exc_av
                logger.warning("AlphaVantage failed with key %s for %s: %s", av_key, symbol, exc_av)
                # try next key
                continue
        logger.warning("All %d AlphaVantage keys tried and failed (last error: %s)", keys_tried, e_av)
    else:
        logger.info("No AlphaVantage keys provided in ALPHA_KEYS; skipping AV attempts.")

    # ---------- 2) Robust yfinance fallback ----------
    try:
        import yfinance as yf
    except Exception as imp_err:
        e_yf = imp_err
        logger.exception("Failed to import yfinance: %s", imp_err)
    else:
        attempts = 3
        backoff = 1.0
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                logger.info("yfinance attempt %d for %s (download)", attempt, symbol)
                # primary: try common download signature (works on newer yfinance)
                try:
                    df_try = yf.download(symbol, period="1y", progress=False, threads=True, timeout=timeout, auto_adjust=False)
                except TypeError:
                    # some yfinance versions don't accept 'threads' or 'timeout' kw in download
                    df_try = yf.download(symbol, period="1y", progress=False, auto_adjust=False)

                if df_try is not None and not df_try.empty:
                    df = df_try
                    logger.info("yfinance.download succeeded for %s (%d rows)", symbol, len(df))
                    break

                # fallback: Ticker.history with fewer kwargs (avoid 'threads' if it errors)
                logger.info("yfinance.download returned empty; trying Ticker.history() for %s", symbol)
                ticker = yf.Ticker(symbol)

                # Try calling history with a safe set of kwargs depending on signature
                hist_func = getattr(ticker, "history", None)
                if hist_func is None:
                    raise RuntimeError("Ticker.history not available in this yfinance version")

                # attempt with commonly-accepted kwargs, fall back to minimal
                try:
                    df_try = ticker.history(period="1y", auto_adjust=False, actions=False)
                except TypeError as te:
                    logger.info("Ticker.history signature mismatch: %s. Retrying with minimal args.", te)
                    df_try = ticker.history(period="1y")

                if df_try is not None and not df_try.empty:
                    df = df_try
                    logger.info("Ticker.history succeeded for %s (%d rows)", symbol, len(df))
                    break

                raise RuntimeError("yfinance returned empty on both download() and Ticker.history()")

            except Exception as e_try:
                last_exc = e_try
                logger.warning("yfinance attempt %d failed for %s: %s", attempt, symbol, e_try)
                time.sleep(backoff)
                backoff *= 2.0
                continue

        if df is None or getattr(df, "empty", True):
            e_yf = last_exc or RuntimeError("yfinance returned empty after retries")
            logger.warning("yfinance fallback failed for %s: %s", symbol, e_yf)

    # If neither AV nor yfinance provided data, try a direct Yahoo JSON endpoint as last resort
    if df is None or getattr(df, "empty", True):
        try:
            # Use a browser-like User-Agent (helps avoid bot blocks)
            yahoo_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1y&interval=1d&includePrePost=false"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                              " Chrome/115.0.0.0 Safari/537.36",
                "Accept": "application/json, text/javascript, */*; q=0.01"
            }
            logger.info("Attempting direct Yahoo chart JSON for %s", symbol)
            resp = requests.get(yahoo_url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()

            # parse JSON -> construct DataFrame
            chart = payload.get("chart", {})
            result = chart.get("result")
            if not result:
                raise RuntimeError(f"Yahoo JSON returned no result: {payload}")

            meta = result[0].get("meta", {})
            timestamps = result[0].get("timestamp", [])
            indicators = result[0].get("indicators", {}).get("quote", [])
            if not timestamps or not indicators or not indicators[0]:
                raise RuntimeError("Yahoo JSON missing timestamps/quote data")

            quote = indicators[0]
            import datetime as _dt
            rows = []
            for i, ts in enumerate(timestamps):
                dt = _dt.datetime.fromtimestamp(ts)
                o = quote.get("open", [None] * len(timestamps))[i]
                h = quote.get("high", [None] * len(timestamps))[i]
                l = quote.get("low", [None] * len(timestamps))[i]
                c = quote.get("close", [None] * len(timestamps))[i]
                v = quote.get("volume", [None] * len(timestamps))[i]
                rows.append({"date": dt, "open": o, "high": h, "low": l, "close": c, "volume": v})

            df_try = pd.DataFrame(rows).set_index("date")
            if df_try is None or df_try.empty:
                raise RuntimeError("Parsed Yahoo JSON produced empty DataFrame")

            df = df_try
            logger.info("Yahoo JSON parsing succeeded for %s (%d rows)", symbol, len(df))

        except Exception as exc_yj:
            e_yahoo_json = exc_yj
            logger.warning("Yahoo JSON fallback failed for %s: %s", symbol, exc_yj)

    # If still empty, raise combined error so caller sees full info
    if df is None or getattr(df, "empty", True):
        logger.error("Data fetch failed for %s. AV error: %s; yfinance error: %s; yahoo_json error: %s",
                     symbol, e_av, e_yf, e_yahoo_json)
        raise RuntimeError(
            f"Data fetch failed for {symbol}. "
            f"AV error: {e_av if e_av is not None else 'n/a'}; "
            f"yfinance error: {e_yf if e_yf is not None else 'n/a'}; "
            f"yahoo_json error: {e_yahoo_json if e_yahoo_json is not None else 'n/a'}"
        )

    # Post-process / normalize and finalize charts
    try:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = _flatten_multiindex_columns(df)
        df = _normalize_columns(df)
        df = df.sort_index()

        alias_pairs = [
            ("1. open", "open"),
            ("2. high", "high"),
            ("3. low", "low"),
            ("4. close", "close"),
            ("5. volume", "volume"),
        ]
        for long_name, short_name in alias_pairs:
            if long_name in df.columns and short_name not in df.columns:
                df[short_name] = df[long_name]
            elif short_name in df.columns and long_name not in df.columns:
                df[long_name] = df[short_name]
        df = df.loc[:, ~df.columns.duplicated()]

        df_final, charts = _finalize_df_and_charts(df)
        return df_final, charts

    except Exception as final_exc:
        logger.exception("Final processing failed for %s: %s", symbol, final_exc)
        raise RuntimeError(f"Final processing failed for {symbol}: {final_exc}")


# ---------- feature builder ----------
def _build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Build X_raw, y_raw and features list using short names:
    features = ["open","high","low","volume","MA10","MA50","RSI","Lag1","Lag2"]
    """
    df = df.copy()
    df = _normalize_columns(df)

    # Ensure 'close' exists and is 1-D
    if "close" not in df.columns:
        raise ValueError("No 'close' column found after normalization.")
    close = df["close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
        df["close"] = close

    # Derived features
    df["MA10"] = df["close"].rolling(10).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["close"])
    df["Lag1"] = df["close"].shift(1)
    df["Lag2"] = df["close"].shift(2)

    df = df.dropna()

    features = ["open", "high", "low", "volume", "MA10", "MA50", "RSI", "Lag1", "Lag2"]

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features after engineering: {missing}. Available: {list(df.columns)}")

    X_raw = df[features].values.astype(float)
    y_raw = df["close"].values.reshape(-1, 1).astype(float)
    return X_raw, y_raw, features


# ---------- prediction logic ----------
def predict_stock_prices_sync(df: pd.DataFrame, days_ahead: int = 1, lookback: int = 15, mc_samples: int = 30):
    """
    Synchronous heavy compute:
      - validate inputs
      - build features, scale, sequence and predict with MC dropout
    Returns (preds:list, confs:list, mae:float, r2:float)
    """
    model, scaler_X, scaler_y = get_model()

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    # normalize and ensure short columns exist
    df = _normalize_columns(df)
    required_cols = ["open", "high", "low", "close", "volume"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c} after normalization. Available: {list(df.columns)}")

    X_raw, y_raw, features = _build_feature_matrix(df)

    if X_raw.shape[0] < (lookback + 10):
        raise ValueError(f"Not enough rows after feature engineering: {X_raw.shape[0]} (need >= {lookback + 10})")

    # scale
    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw)

    # build sequences
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i])
        y_seq.append(y_scaled[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if len(X_seq) < 10:
        raise ValueError("Not enough sequences to evaluate (need >= 10).")

    # train/test split for metrics
    split = int(0.8 * len(X_seq))
    X_test = X_seq[split:]
    y_test = y_seq[split:]

    # model predict on test set (robust)
    try:
        y_test_pred_scaled = model.predict(X_test)
    except Exception as e:
        try:
            y_test_pred_scaled = model(X_test, training=False).numpy()
        except Exception:
            raise RuntimeError(f"Model prediction on X_test failed: {e}")

    y_test_pred_scaled = np.asarray(y_test_pred_scaled).reshape(-1, 1)
    y_test_actual_scaled = np.asarray(y_test).reshape(-1, 1)

    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test_actual_scaled)
    mae = float(np.mean(np.abs(y_test_actual - y_test_pred)))
    # r2_score may fail on degenerate data; compute safely
    try:
        from sklearn.metrics import r2_score
        r2 = float(r2_score(y_test_actual, y_test_pred))
    except Exception:
        r2 = 0.0

    # Forecasting with MC Dropout
    raw_closes = list(y_raw.ravel())
    last_raw_window = X_raw[-lookback:].copy()
    last_scaled_window = X_scaled[-lookback:].copy()

    preds = []
    confs = []

    # indices within feature vector (using features list returned earlier)
    lag1_idx = features.index("Lag1")
    lag2_idx = features.index("Lag2")
    volume_idx = features.index("volume")
    ma10_idx = features.index("MA10")
    ma50_idx = features.index("MA50")
    rsi_idx = features.index("RSI")

    for step in range(days_ahead):
        mc_prices = []
        for _ in range(mc_samples):
            try:
                out = model(last_scaled_window[np.newaxis, :, :], training=True)
                pred_scaled = np.asarray(out).reshape(-1, 1)[0, 0]
            except Exception:
                out = model.predict(last_scaled_window[np.newaxis, :, :])
                pred_scaled = np.asarray(out).reshape(-1, 1)[0, 0]
            price = float(scaler_y.inverse_transform([[pred_scaled]])[0][0])
            mc_prices.append(price)

        mc_prices = np.array(mc_prices)
        mean_price = float(mc_prices.mean())
        std_price = float(mc_prices.std())

        preds.append(round(mean_price, 4))
        if mean_price <= 0:
            conf = 50.0
        else:
            conf = 100.0 - (std_price / max(mean_price, 1e-9) * 100.0)
            conf = float(np.clip(conf, 1.0, 99.9))
        confs.append(round(conf, 2))

        # update raw_closes + build next raw feature vector
        raw_closes.append(mean_price)
        recent_closes = pd.Series(raw_closes[-50:])
        ma10 = float(recent_closes.tail(10).mean()) if len(recent_closes) >= 10 else float(recent_closes.mean())
        ma50 = float(recent_closes.tail(50).mean()) if len(recent_closes) >= 50 else float(recent_closes.mean())
        rsi_series = calculate_rsi(recent_closes, period=14)
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else 50.0

        prev_raw = last_raw_window[-1].copy()
        new_raw = prev_raw.copy()
        # approximate OHLC by predicted price (heuristic)
        new_raw[0] = mean_price  # open
        new_raw[1] = mean_price  # high
        new_raw[2] = mean_price  # low
        # keep volume same as previous
        new_raw[volume_idx] = prev_raw[volume_idx]
        new_raw[ma10_idx] = ma10
        new_raw[ma50_idx] = ma50
        new_raw[rsi_idx] = rsi_val

        # compute latest lag values
        prev_close = raw_closes[-2] if len(raw_closes) >= 2 else mean_price
        prev_prev_close = raw_closes[-3] if len(raw_closes) >= 3 else prev_close
        new_raw[lag2_idx] = prev_prev_close
        new_raw[lag1_idx] = prev_close

        # roll windows
        last_raw_window = np.vstack([last_raw_window[1:], new_raw])
        new_scaled_row = scaler_X.transform(new_raw.reshape(1, -1))[0]
        last_scaled_window = np.vstack([last_scaled_window[1:], new_scaled_row])

    preds_out = [float(p) for p in preds]
    confs_out = [float(c) for c in confs]
    return preds_out, confs_out, float(mae), float(r2)

# def fetch_news_urls(company_name, num_articles=5):
#     """
#     Fetch news URLs for a company using googlesearch.
#     Works with older 'googlesearch' (no stop) or googlesearch-python (with num_results).
#     """
#     try:
#         from googlesearch import search
#     except ImportError:
#         print("Please install googlesearch-python: pip install googlesearch-python")
#         return []

#     query = f"{company_name} stock news"
#     urls = []

#     try:
#         # Try googlesearch-python signature
#         for url in search(query, num_results=num_articles):
#             urls.append(url)
#     except TypeError:
#         # Fallback for old googlesearch signature
#         for i, url in enumerate(search(query, num=num_articles, pause=2)):
#             urls.append(url)
#             if i + 1 >= num_articles:
#                 break

#     return urls


# def extract_article(url):
#     try:
#         article = Article(url)
#         article.download()
#         article.parse()
#         article.nlp()
#         return {
#             "title": article.title,
#             "text": article.text,
#             "summary": getattr(article, "summary", ""),
#             "source": article.source_url if hasattr(article, "source_url") else "",
#             "published": article.publish_date.strftime("%Y-%m-%d %H:%M") if article.publish_date else None,
#             "url": url,
#         }
#     except Exception as e:
#         logger.warning("Failed to extract article from %s: %s", url, e)
#         return {"url": url, "error": str(e)}  # 👈 return debug info instead of None


# def analyze_sentiment(text):
#     analyzer = SentimentIntensityAnalyzer()
#     return analyzer.polarity_scores(text)  # returns dict with pos, neg, neu, compound

# def company_sentiment(company_name, num_articles=5):
#     urls = fetch_news_urls(company_name, num_articles)

#     print("Fetched URLs:", urls)  # Debug: show fetched URLs

#     all_text = ""

#     for url in urls:
#         art = extract_article(url)
#         if not art or not art["text"]:
#             continue
#         print("\nArticle text preview:", art["text"][:200])
#         all_text += art["text"] + "\n"

#     if not all_text.strip():
#         print("No text extracted from articles.")
#         return None  # no news found

#     sentiment = analyze_sentiment(all_text)
#     sentiment['date'] = datetime.now().strftime("%Y-%m-%d")
#     sentiment['company'] = company_name
#     return sentiment


# def save_sentiment_trend(company_name, num_articles=5, csv_file="sentiment_trend.csv"):
#     sentiment = company_sentiment(company_name, num_articles)
#     if sentiment:
#         try:
#             df = pd.read_csv(csv_file)
#         except FileNotFoundError:
#             df = pd.DataFrame()

#         df = pd.concat([df, pd.DataFrame([sentiment])], ignore_index=True)
#         df.to_csv(csv_file, index=False)
#         print(f"Saved sentiment for {company_name} on {sentiment['date']}")
#     else:
#         print("No news found today.")

# ---------- routes ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Stock Price Prediction (LSTM) - Debuggable</h1>
    <p>GET /predict?stock_symbol=AAPL&days_ahead=5</p>
    """

# ---------- route: /sentiment (formatted to match your TSX shape) ----------
# @app.get("/sentiment")
# def sentiment_route(company: str = Query(...), num_articles: int = 5, debug: bool = False):
#     try:
#         urls = fetch_news_urls(company, num_articles)
#         logger.info("Fetched URLs for %s: %s", company, urls)

#         if not urls:
#             return JSONResponse(
#                 {"error": f"No articles found for {company}"},
#                 status_code=404,
#             )

#         results = []
#         for url in urls:
#             try:
#                 art = extract_article(url)
#                 if not art or not art["text"]:
#                     logger.warning("Skipped article %s due to missing text. Error: %s", art.get("url"), art.get("error"))
#                     continue

#                 logger.info("Analyzing article: %s", art["title"])
#                 sentiment_scores = analyze_sentiment(art["text"])
#                 compound = sentiment_scores.get("compound", 0.0)
#                 confidence = abs(compound) * 100

#                 impact = "high" if confidence > 70 else "medium" if confidence > 40 else "low"

#                 results.append({
#                     "title": art["title"],
#                     "summary": art["summary"],
#                     "source": art["source"],
#                     "time": art["published"] or datetime.now().strftime("%Y-%m-%d %H:%M"),
#                     "sentiment": sentiment_scores,
#                     "impact": impact,
#                     "aiConfidence": round(confidence, 2),
#                     "url": art["url"]
#                 })
#             except Exception as e:
#                 tb = traceback.format_exc()
#                 logger.error("Error processing article %s: %s\n%s", url, e, tb)
#                 if debug:
#                     results.append({
#                         "error": str(e),
#                         "traceback": tb,
#                         "url": url
#                     })
#                 continue

#         if not results:
#             return JSONResponse(
#                 {"error": f"Articles fetched but none could be parsed for {company}"},
#                 status_code=502,
#             )

#         return results

#     except Exception as exc:
#         tb = traceback.format_exc()
#         logger.error("Sentiment route error: %s\n%s", exc, tb)
#         return JSONResponse(
#             {"error": str(exc), "traceback": tb if debug else None},
#             status_code=500,
        # )
# def sentiment_route(company: str = Query(...), num_articles: int = 5):
#     urls = fetch_news_urls(company, num_articles)
#     if not urls:
#         return JSONResponse({"error": "No articles found"}, status_code=404)

#     results = []
#     for url in urls:
#         art = extract_article(url)
#         if not art or not art["text"]:
#             continue

#         sentiment_scores = analyze_sentiment(art["text"])  # dict
#         compound = sentiment_scores["compound"]
#         confidence = abs(compound) * 100  # e.g., derive % confidence

#         impact = "high" if confidence > 70 else "medium" if confidence > 40 else "low"

#         results.append({
#             "title": art["title"],
#             "summary": art["summary"],
#             "source": art["source"],
#             "time": art["published"] or datetime.now().strftime("%Y-%m-%d %H:%M"),
#             "sentiment": sentiment_scores,
#             "impact": impact,
#             "aiConfidence": round(confidence, 2),
#             "url": art["url"]
#         })

#     return results


@app.get("/predict")
async def predict_endpoint(
    stock_symbol: str = Query(...),
    outputsize: str = Query("compact"),
    timeout: int = 10,
    days_ahead: int = Query(5),
    lookback: int = Query(15, ge=5, le=200),
    mc_samples: int = Query(30, ge=1, le=200),
    debug: Optional[bool] = False,
):
    """
    Async endpoint: fetches data, runs predict in threadpool, returns JSON.
    """
    try:
        logger.info(
            "Predict request for %s days=%s lookback=%s mc=%s, Fetch timeout=%s",
            stock_symbol,
            days_ahead,
            lookback,
            mc_samples,
            timeout,
        )
        df, charts = await run_in_threadpool(fetch_stock_data, stock_symbol, outputsize, timeout)
        logger.info("Fetched df with %d rows for %s", len(df), stock_symbol)

        preds, confs, mae, r2 = await run_in_threadpool(
            predict_stock_prices_sync, df, days_ahead, lookback, mc_samples
        )

        payload = {
            "stock_symbol": stock_symbol,
            "days_ahead": int(days_ahead),
            "predictions": preds,
            "confidence": confs,
            "mae": mae,
            "r2": r2,
            "price_data": charts.get("price_data", []),
            "volume_data": charts.get("volume_data", []),
            "rsi_data": charts.get("rsi_data", []),
        }
        return JSONResponse(content=payload)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Predict or Fetch error: %s\n%s", exc, tb)
        return JSONResponse(content={"error": str(exc), "traceback": tb if debug else None}, status_code=500)


# ---------- run ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting on port %s", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
