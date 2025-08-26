# main.py — improved, debug-friendly FastAPI app
import io
import os
import json
import traceback
import logging
from typing import Optional, Tuple, List

import requests
import numpy as np
import pandas as pd
import joblib

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# ML / TF
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------- config ----------
load_dotenv()
API_KEY = os.getenv("MY_SECRET_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", API_KEY)
AV_BASE_URL = "https://www.alphavantage.co/query"
MOD_DIR = "mod"
# MODEL_KERAS_PATH = os.path.join(MOD_DIR, "model.keras")  # or .h5
# SCALER_X_PATH = os.path.join(MOD_DIR, "scaler_X.pkl")
# SCALER_Y_PATH = os.path.join(MOD_DIR, "scaler_y.pkl")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-predict")


# FastAPI
app = FastAPI(title="📈 Stock Price Prediction API - Debuggable",
              description="Predict stock prices using a trained BiLSTM model (inference-only).",
              version="2.1.0")

origins = [
    "https://market-trend-analysis.vercel.app",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"])

# @app.get("/predict")
# async def predict_endpoint(
#     stock_symbol: str = Query(..., min_length=1),
#     days_ahead: int = Query(5, ge=1, le=30),
#     lookback: int = Query(15, ge=5, le=200),
#     mc_samples: int = Query(30, ge=1, le=200),
#     debug: Optional[bool] = False
# ):
#     """
#     GET /predict?stock_symbol=TSLA&days_ahead=5&lookback=15&mc_samples=30
#     """
#     try:
#         logger.info("Predict request for %s days=%s lookback=%s mc=%s", stock_symbol, days_ahead, lookback, mc_samples)

#         # fetch historical data (runs in threadpool so event loop doesn't block)
#         df = await run_in_threadpool(fetch_stock_data, stock_symbol, "full", 10)
#         logger.info("Fetched df: rows=%d for %s", len(df), stock_symbol)

#         # run heavy compute (train/predict/forecast) in threadpool
#         preds, confs, mae, r2 = await run_in_threadpool(
#             predict_stock_prices_sync, df, days_ahead, lookback, mc_samples
#         )

#         payload = {
#             "stock_symbol": stock_symbol.upper(),
#             "days_ahead": int(days_ahead),
#             "lookback": int(lookback),
#             "mc_samples": int(mc_samples),
#             "predictions": [float(x) for x in preds],
#             "confidence": [float(x) for x in confs],
#             "mae": float(mae),
#             "r2": float(r2),
#         }
#         return JSONResponse(content=payload)
#     except Exception as exc:
#         tb = traceback.format_exc()
#         logger.error("Predict error: %s\n%s", exc, tb)
#         return JSONResponse(content={"error": str(exc), "traceback": tb if debug else None}, status_code=500)

# sentiment= joblib.load("mod/sentiment_model.pkl")
sentiment = None

# @app.on_event("startup")
# def load_models():
#     global sentiment
#     try:
#         sentiment = joblib.load("mod/sentiment_model.pkl")
#         logger.info("Sentiment model loaded")
#     except Exception:
#         sentiment = None
#         logger.exception("Could not load sentiment model at startup")


# ---------- utilities ----------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    """Read CSV using requests to control timeouts and raise nice errors."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


# ---------- model & scaler loading at startup ----------
# @app.on_event("startup")
# def load_artifacts():
#     global model, scaler_X, scaler_y
    # Model
    # try:
    #     if os.path.exists(MODEL_KERAS_PATH):
    #         logger.info("Loading Keras model from %s", MODEL_KERAS_PATH)
    #         model = load_model(MODEL_KERAS_PATH)
    #         logger.info("Model loaded successfully.")
    #     else:
    #         logger.error("Model file not found at %s", MODEL_KERAS_PATH)
    #         model = None
    # except Exception:
    #     logger.exception("Failed to load model. Trace:")
    #     model = None

    # Scalers
    # try:
    #     if os.path.exists(SCALER_X_PATH):
    #         scaler_X = joblib.load(SCALER_X_PATH)
    #         logger.info("Loaded scaler_X from %s", SCALER_X_PATH)
    #     else:
    #         logger.error("scaler_X not found at %s", SCALER_X_PATH)
    #         scaler_X = None
    # except Exception:
    #     logger.exception("Failed to load scaler_X. Trace:")
    #     scaler_X = None

    # try:
    #     if os.path.exists(SCALER_Y_PATH):
    #         scaler_y = joblib.load(SCALER_Y_PATH)
    #         logger.info("Loaded scaler_y from %s", SCALER_Y_PATH)
    #     else:
    #         logger.error("scaler_y not found at %s", SCALER_Y_PATH)
    #         scaler_y = None
    # except Exception:
    #     logger.exception("Failed to load scaler_y. Trace:")
    #     scaler_y = None


# ---------- data fetcher ----------
def fetch_stock_data(symbol: str, outputsize: str = "compact", timeout: int = 10) -> pd.DataFrame:
    """Try AlphaVantage CSV first, fallback to yfinance. Return normalized DataFrame with datetime index."""
    if ALPHA_VANTAGE_API_KEY is None:
        logger.warning("ALPHA_VANTAGE_API_KEY not set; yfinance fallback will be attempted if available.")
    e_av = None
    # Try Alpha Vantage CSV
    try:
        url = (
            f"{AV_BASE_URL}?function=TIME_SERIES_DAILY&symbol={symbol}"
            f"&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}&datatype=csv"
        )
        logger.info("Fetching CSV from AlphaVantage: %s", url)
        df = safe_read_csv_from_url(url, timeout=timeout)
        # set index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        df = _normalize_columns(df)
        df = df.sort_index()
        if df.empty:
            raise ValueError("AlphaVantage returned empty CSV")
        return df
    except Exception as e_av:
        logger.warning("AlphaVantage CSV failed for %s: %s", symbol, e_av)

    # Fallback: yfinance
    try:
        import yfinance as yf
        logger.info("Falling back to yfinance for symbol %s", symbol)
        df = yf.download(symbol, period="2y", progress=False)
        if df is None or df.empty:
            raise ValueError("yfinance returned empty")
        df = df.rename(columns={
            "Open": "1. open",
            "High": "2. high",
            "Low": "3. low",
            "Close": "4. close",
            "Volume": "5. volume",
            "Adj Close": "adjclose"
        })
        df.index = pd.to_datetime(df.index)
        df = _normalize_columns(df)
        df = df.sort_index()
        return df
    except Exception as e_yf:
        logger.exception("yfinance fallback failed for %s: %s", symbol, e_yf)
        raise RuntimeError(f"Data fetch failed for {symbol}. AV error: {e_av if 'e_av' in locals() else 'n/a'}; yfinance error: {e_yf}")

def fetch_data(symbol: str, outputsize: str = "compact", timeout: int = 10):
    """Fetch stock data (AlphaVantage -> yfinance fallback) and return formatted dicts for charts."""

    # STEP 1: Fetch raw data (same as your existing code)
    try:
        url = (
            f"{AV_BASE_URL}?function=TIME_SERIES_DAILY&symbol={symbol}"
            f"&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}&datatype=csv"
        )
        logger.info("Fetching CSV from AlphaVantage: %s", url)
        df = safe_read_csv_from_url(url, timeout=timeout)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        df = _normalize_columns(df)
        df = df.sort_index()
        if df.empty:
            raise ValueError("AlphaVantage returned empty CSV")

    except Exception as e_av:
        logger.warning("AlphaVantage failed for %s: %s", symbol, e_av)
        import yfinance as yf
        df = yf.download(symbol, period="2y", progress=False)
        if df is None or df.empty:
             return {"price_data": [], "volume_data": [], "rsi_data": []}
        # Normalize columns
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjclose",
            "Volume": "volume"
        })
        df.index = pd.to_datetime(df.index)
        df = _normalize_columns(df)
        df = df.sort_index()
        # Defensive: ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return {"price_data": [], "volume_data": [], "rsi_data": []}
    # STEP 2: Calculate indicators
    df["ma20"] = df["4. close"].rolling(window=20).mean()
    df["ma50"] = df["4. close"].rolling(window=50).mean()
    df["rsi"] = calculate_rsi(df["4. close"])

    # STEP 3: Format output as lists of dicts
    df_monthly = df.resample("M").last()  # monthly snapshot for chart

    price_data = [
        {
            "date": d.strftime("%b %Y"),
            "close": round(row["4. close"], 2),
            "ma20": round(row["ma20"], 2) if not pd.isna(row["ma20"]) else None,
            "ma50": round(row["ma50"], 2) if not pd.isna(row["ma50"]) else None
        }
        for d, row in df_monthly.iterrows()
    ]

    volume_data = [
        {"date": d.strftime("%b %Y"), "volume": int(row["5. volume"])}
        for d, row in df_monthly.iterrows()
    ]

    rsi_data = [
        {"date": d.strftime("%b %Y"), "rsi": round(row["rsi"], 2) if not pd.isna(row["rsi"]) else None}
        for d, row in df_monthly.iterrows()
    ]

    return {
        "price_data": price_data,
        "volume_data": volume_data,
        "rsi_data": rsi_data
    }

# ---------- prediction logic (synchronous heavy work) ----------
def _build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """Return (X_raw, y_raw, features_list)."""
    df = df.copy()
    df["MA10"] = df["4. close"].rolling(10).mean()
    df["MA50"] = df["4. close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["4. close"])
    df["Lag1"] = df["4. close"].shift(1)
    df["Lag2"] = df["4. close"].shift(2)
    df = df.dropna()
    features = ["1. open", "2. high", "3. low", "5. volume", "MA10", "MA50", "RSI", "Lag1", "Lag2"]
    X_raw = df[features].values.astype(float)
    y_raw = df["4. close"].values.reshape(-1, 1).astype(float)
    return X_raw, y_raw, features


def predict_stock_prices_sync(df: pd.DataFrame, days_ahead: int = 1, lookback: int = 15, mc_samples: int = 30):
    """
    Synchronous, CPU/TensorFlow-heavy function that:
      - validates inputs
      - uses preloaded model & scalers (from module globals)
      - computes test metrics and MC-dropout forecasts
    Returns: (preds:list, confidences:list, mae:float, r2:float)
    Raises: informative exceptions on error
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    # if model is None:
    #     raise RuntimeError("Model not loaded (see server logs).")
    # if scaler_X is None or scaler_y is None:
    #     raise RuntimeError("Scalers not loaded (see server logs).")

    # normalize & build features
    df = _normalize_columns(df)
    required_cols = ["1. open", "2. high", "3. low", "4. close", "5. volume"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c} after normalization. Available: {list(df.columns)}")

    X_raw, y_raw, features = _build_feature_matrix(df)

    if X_raw.shape[0] < (lookback + 10):
        raise ValueError(f"Not enough rows after feature engineering: {X_raw.shape[0]} (need >= {lookback + 10})")

    # Scale using saved scalers (transform only!)
    scaler_X= MinMaxScaler()
    scaler_y= MinMaxScaler()
    # scaler_y= MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_raw)   # IMPORTANT: transform, not fit_transform
    y_scaled = scaler_y.fit_transform(y_raw)

    # Build sequences
    X_seq, y_seq = [], []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i])
        y_seq.append(y_scaled[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if len(X_seq) < 10:
        raise ValueError("Not enough sequences to evaluate (need >= 10).")

    # train/test split (for metrics)
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

 # --- Build LSTM Model ---
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(lookback, X_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)
    # Evaluate on test set
    try:
        y_test_pred_scaled = model.predict(X_test)
    except Exception as e:
        # fallback to calling model(...) if predict fails (rare)
        try:
            y_test_pred_scaled = model(X_test, training=False).numpy()
        except Exception:
            raise RuntimeError(f"Model prediction on X_test failed: {e}")

    y_test_pred_scaled = np.asarray(y_test_pred_scaled).reshape(-1, 1)
    y_test_actual_scaled = np.asarray(y_test).reshape(-1, 1)

    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test_actual_scaled)
    mae = float(mean_absolute_error(y_test_actual, y_test_pred))
    r2 = float(r2_score(y_test_actual, y_test_pred))

    # Forecasting with MC Dropout (maintain raw feature window and recalc derived features)
    # Prepare raw close series for recalculating MA/RSI
    raw_closes = list(y_raw.ravel())  # chronological closes
    last_raw_window = X_raw[-lookback:].copy()  # raw features for last lookback entries
    last_scaled_window = X_scaled[-lookback:].copy()

    preds = []
    confs = []

    # helper indices
    lag1_idx = features.index("Lag1")
    lag2_idx = features.index("Lag2")
    volume_idx = features.index("5. volume")

    for step in range(days_ahead):
        mc_prices = []
        mc_scaled = []
        for _ in range(mc_samples):
            # attempt stochastic forward pass (training=True) — works only if model is Keras and contains Dropout
            try:
                out = model(last_scaled_window[np.newaxis, :, :], training=True)
                pred_scaled = np.asarray(out).reshape(-1, 1)[0, 0]
            except Exception:
                # fallback to deterministic predict
                out = model.predict(last_scaled_window[np.newaxis, :, :])
                pred_scaled = np.asarray(out).reshape(-1, 1)[0, 0]
            mc_scaled.append(pred_scaled)
            # inverse to raw price
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

        # update raw_closes and last_raw_window
        raw_closes.append(mean_price)  # append predicted raw close
        # compute new derived features based on updated close series
        recent_closes = pd.Series(raw_closes[-50:])  # last up to 50 closes for MA50
        ma10 = float(recent_closes.tail(10).mean()) if len(recent_closes) >= 10 else float(recent_closes.mean())
        ma50 = float(recent_closes.tail(50).mean()) if len(recent_closes) >= 50 else float(recent_closes.mean())
        rsi_series = calculate_rsi(recent_closes, period=14)
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else 50.0

        # build new raw feature vector:
        prev_raw = last_raw_window[-1].copy()
        new_raw = prev_raw.copy()
        # approximate open/high/low -> use prev close as placeholder (heuristic)
        new_raw[0] = mean_price  # "1. open"
        new_raw[1] = mean_price  # "2. high"
        new_raw[2] = mean_price  # "3. low"
        # volume: use last known volume as placeholder
        new_raw[volume_idx] = prev_raw[volume_idx]
        new_raw[features.index("MA10")] = ma10
        new_raw[features.index("MA50")] = ma50
        new_raw[features.index("RSI")] = rsi_val
        # Lag2 <- previous Lag1 (which is previous close), Lag1 <- previous close (which was last element of raw_closes before append)
        prev_close = raw_closes[-2] if len(raw_closes) >= 2 else mean_price
        prev_prev_close = raw_closes[-3] if len(raw_closes) >= 3 else prev_close
        new_raw[lag2_idx] = prev_prev_close
        new_raw[lag1_idx] = prev_close

        # shift windows: raw then scaled
        last_raw_window = np.vstack([last_raw_window[1:], new_raw])
        new_scaled_row = scaler_X.transform(new_raw.reshape(1, -1))[0]
        last_scaled_window = np.vstack([last_scaled_window[1:], new_scaled_row])

    # ensure all outputs are JSON-serializable native python types
    preds_out = [float(p) for p in preds]
    confs_out = [float(c) for c in confs]
    return preds_out, confs_out, mae, r2


# ---------- routes ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Stock Price Prediction (LSTM) - Debuggable</h1>
    <p>GET /predict?stock_symbol=AAPL&days_ahead=5&debug=true</p>
    """

@app.get("/fetch")
async def fetch(stock_symbol: str = Query(...),
                           outputsize: str = Query("compact"),
                           timeout: int=10,
                           debug: Optional[bool] = False):

    try:
        logger.info("Fetch request for %s timeout=%s", stock_symbol, timeout)

        result= await run_in_threadpool(fetch_data, stock_symbol,outputsize, timeout)
        # result is a dict
        price_data = result.get("price_data") or []
        volume_data = result.get("volume_data") or []
        rsi_data = result.get("rsi_data") or []
        payload = {
            "price_data": price_data,
            "volume_data": volume_data,
            "rsi_data": rsi_data
        }
        return JSONResponse(content=payload)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Fetch error: %s\n%s", exc, tb)
        return JSONResponse(content={"error": str(exc), "traceback": tb if debug else None}, status_code=500)
    
@app.get("/predict")
async def predict_endpoint(stock_symbol: str = Query(...),
                           days_ahead: int = Query(5),
                            lookback: int = Query(15, ge=5, le=200),
                            mc_samples: int = Query(30, ge=1, le=200),
                           debug: Optional[bool] = False):
    """
    Async endpoint: fetches data, runs predict in threadpool, returns JSON.
    If debug=True the traceback will be included in the response.
    """
    try:
        logger.info("Predict request for %s days=%s lookback=%s mc=%s", stock_symbol, days_ahead, lookback, mc_samples)
        df = await run_in_threadpool(fetch_stock_data, stock_symbol, "full", 10)
        logger.info("Fetched df with %d rows for %s", len(df), stock_symbol)

        # run heavy predict in threadpool
        preds, confs, mae, r2 = await run_in_threadpool(predict_stock_prices_sync, df, days_ahead, lookback, mc_samples)

        payload = {
            "stock_symbol": stock_symbol,
            "days_ahead": int(days_ahead),
            "predictions": preds,
            "confidence": confs,
            "mae": float(mae),
            "r2": float(r2),
        }
        return JSONResponse(content=payload)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Predict error: %s\n%s", exc, tb)
        return JSONResponse(content={"error": str(exc), "traceback": tb if debug else None}, status_code=500)


# ---------- run ----------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting on port %s", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
