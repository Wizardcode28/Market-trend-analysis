# main.py — cleaned, defensive, AlphaVantage-first (with robust yfinance fallback) and scaler/bundle handling
from fastapi import FastAPI, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import requests
from typing import List, Optional, Dict
import logging
import traceback
import joblib
import json
import pickle
import re
i am totally frustated and i want to make main.py from scratch again, this is description of all things
model.pth, actually there is no model for just predicting only probability of going up and down, we have two models, one for predicting stock prices which uses now LSTM model instead of earlier randomforest and other one is for predicting sentiment for news which i will handle later now i only care about prices prediction using LSTM, i have uploaded file, first  



What it contains: Only the learned parameters (weights and biases).

How it’s used: You load it into a model class you define yourself.


import torch
from my_model_def import MyModelClass

model = MyModelClass()
model.load_state_dict(torch.load("model.pth"))
model.eval()

Important: You need to know or recreate the architecture (i.e., the MyModelClass) that matches this state dict.

 bundle.pkl (or Bundle.pkl)

What it is: A Python pickle file (.pkl) — usually created during model training or export.

What it might contain:

Model architecture/configuration (so you don't need to manually define the model class)

Tokenizers or preprocessing logic

Metadata (e.g. input/output shapes, version info, label maps)

Training configuration (e.g., optimizer settings, hyperparameters)


How it’s used:

Depends on the framework — for example, TorchServe, FastAI, or other ML tooling may automatically consume bundle.pkl to reconstruct everything needed to serve or evaluate the 


model.pth: Just the weights. You need the model class to use it.

bundle.pkl: Contains the context to interpret and use the model — possibly even enough to recreate the model class dynamically.
# Optional libs
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except Exception:
    ort = None
    ORT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

# yfinance optional (used only as fallback)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    yf = None
    YFINANCE_AVAILABLE = False

# ----------------------
# Config / defaults (from training script)
# ----------------------
FALLBACK_FEATURES = ['open','high','low','close','adjclose','volume','rsiadjclose15','rsivolume15','rsiad']
FALLBACK_LOOKBACK = 30
FALLBACK_MODEL_KW = {"input_dim": len(FALLBACK_FEATURES), "hidden_dim":64, "num_layers":2, "dropout":0.3}

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", os.getenv("MY_SECRET_API_KEY"))
AV_BASE_URL = 'https://www.alphavantage.co/query'

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stock-predict")
logger.setLevel(logging.DEBUG)

# ----------------------
# LSTM class (only if torch available) - matches training LSTM
# ----------------------
if TORCH_AVAILABLE:
    class LSTMClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return self.sigmoid(out).squeeze(-1)
else:
    class LSTMClassifier:
        def __init__(self, *a, **k):
            raise RuntimeError("PyTorch is not available in this environment.")

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI(title="Stock Direction API", version="1.0 (LSTM classifier)")
origins = ["http://localhost:8080", "http://127.0.0.1:8080"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET","POST"], allow_headers=["*"])

# ----------------------
# Helpers: RSI, feature prep, fetch data
# ----------------------
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

import re
# ... (other imports)

def _normalize_colname(col: str) -> str:
    # remove punctuation/spaces, keep digits/letters
    key = re.sub(r'[^0-9a-z]', '', str(col).lower())
    return key

def prepare_features_for_bundle(df_raw: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Robustly normalize column names and produce dataframe with required features.
    Raises ValueError listing available columns if required features are missing.
    """
    if df_raw is None or len(df_raw.columns) == 0:
        raise ValueError("prepare_features_for_bundle: empty dataframe provided")

    df = df_raw.copy()
    # keep original columns for debug
    orig_cols = list(df.columns)

    # Build mapping from normalized -> original column
    norm_map = {}
    for c in df.columns:
        norm = _normalize_colname(c)
        norm_map[norm] = c

    # create canonical columns if we can find them
    canonical = {}
    # candidates for each canonical name (normalized)
    candidates = {
        "open": ["1open", "open"],
        "high": ["2high", "high"],
        "low":  ["3low", "low"],
        "close":["4close", "close"],
        "adjclose": ["5adjustedclose", "5adjustedclose", "5adjusted_close", "adjclose", "adjclose"],
        "volume": ["6volume", "volume"]
    }
    for canon, pats in candidates.items():
        found = None
        for p in pats:
            if p in norm_map:
                found = norm_map[p]
                break
        if found:
            canonical[canon] = found

    # if adjclose missing, fall back to close if present
    if "adjclose" not in canonical and "close" in canonical:
        canonical["adjclose"] = canonical["close"]

    # rename df columns to canonical where possible
    rename_map = {orig: canon for canon, orig in canonical.items()}
    df = df.rename(columns=rename_map)

    # lower-case and strip columns to final canonical names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # now check & compute engineered features requested
    features_clean = [str(f).strip().lower() for f in features]

    # compute RSI features if requested
    if any(f.startswith("rsi") for f in features_clean):
        if "adjclose" in df.columns:
            if "rsiadjclose15" in features_clean and "rsiadjclose15" not in df.columns:
                df["rsiadjclose15"] = calculate_rsi(df["adjclose"].astype(float), period=15)
            if "rsiad" in features_clean and "rsiad" not in df.columns:
                df["rsiad"] = calculate_rsi(df["adjclose"].astype(float), period=14)
        if "volume" in df.columns:
            if "rsivolume15" in features_clean and "rsivolume15" not in df.columns:
                df["rsivolume15"] = calculate_rsi(df["volume"].astype(float), period=15)

    # moving averages
    if "ma20" in features_clean and "ma20" not in df.columns and "close" in df.columns:
        df["ma20"] = df["close"].rolling(20).mean()
    if "ma50" in features_clean and "ma50" not in df.columns and "close" in df.columns:
        df["ma50"] = df["close"].rolling(50).mean()

    # make sure all requested features exist now
    missing = [f for f in features_clean if f not in df.columns]
    if missing:
        logger.debug("Original columns: %s", orig_cols)
        logger.debug("Normalized/renamed columns now: %s", list(df.columns))
        raise ValueError(f"Missing features required by model: {missing}. Available columns: {list(df.columns)[:50]}")

    df_feat = df[features_clean].dropna().copy()
    if df_feat.empty:
        raise ValueError("Feature DataFrame is empty after dropna; check data completeness.")
    return df_feat

def fetch_stock_data(stock_symbol: str, period: str = "1y", timeout: int = 10) -> pd.DataFrame:
    """
    Alpha Vantage primary (keeps your requested order). If AV fails,
    try multiple yfinance fallbacks and return a detailed error on failure.
    """
    logger.info("Fetching stock data for %s (Alpha Vantage primary)", stock_symbol)

    av_err_text = None
    # --- 1) Try Alpha Vantage first (primary) ---
    try:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": stock_symbol,
            "outputsize": "full",   # full as in your earlier working code
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        resp = requests.get(AV_BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # log the raw AV response for debugging (only first 500 chars)
        logger.debug("AlphaVantage raw response (truncated): %s", str(data)[:500])

        if isinstance(data, dict) and "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            # keep AV-like column names for consistency with the rest of your code
            # convert to numeric where possible (coerce errors)
            df = df.apply(pd.to_numeric, errors="coerce")
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.sort_index()
            if df.empty:
                raise ValueError("Alpha Vantage returned empty DataFrame")
            logger.info("AlphaVantage success for %s, rows=%d", stock_symbol, len(df))
            return df
        else:
            # Return specific Note / Error Message if present
            av_err_text = (data.get("Note") or data.get("Error Message") or repr(data))
            raise ValueError(f"Alpha Vantage error: {av_err_text}")

    except Exception as e_av:
        # capture AV exception string for returning to user
        logger.warning("AlphaVantage failed for %s: %s", stock_symbol, e_av)
        # keep av_err_text if set, else fallback to exception message
        if av_err_text is None:
            av_err_text = str(e_av)

    # --- 2) Try yfinance fallbacks (only if available) ---
    y_attempts = []
    last_yf_error = None
    if YFINANCE_AVAILABLE:
        # build candidate ticker variants (defensive)
        candidates = []
        base = stock_symbol
        candidates.append(stock_symbol)
        if "." in stock_symbol:
            base = stock_symbol.split(".")[0]
            candidates.append(base)
        # ensure common NSE suffix
        if not stock_symbol.endswith(".NS"):
            candidates.append(base + ".NS")
        # remove duplicates while preserving order
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        for sym in candidates:
            try:
                logger.debug("yfinance trying %s (period=%s)", sym, period)
                # First try yf.download
                df = yf.download(sym, period=period, progress=False)
                if df is not None and not df.empty:
                    # normalize names to AV-style so rest of pipeline works
                    # df = df.rename(columns={
                    #     "Open": "1. open",
                    #     "High": "2. high",
                    #     "Low": "3. low",
                    #     "Close": "4. close",
                    #     "Adj Close": "5. adjusted close",
                    #     "Volume": "6. volume"
                    # })
                    df = df.rename(columns={
                "('1. open', 'tcs.ns')": "open",
                "('2. high', 'tcs.ns')": "high",
                "('3. low', 'tcs.ns')": "low",
                "('4. close', 'tcs.ns')": "close",
                "('6. volume', 'tcs.ns')": "volume"
            })
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    logger.info("yfinance success for %s (download) rows=%d", sym, len(df))
                    return df
                # If download returned empty, try yf.Ticker.history as a secondary method
                logger.debug("yfinance download empty for %s, trying Ticker.history()", sym)
                t = yf.Ticker(sym)
                df2 = t.history(period=period)
                if df2 is not None and not df2.empty:
                    df2 = df2.rename(columns={
                        "Open": "1. open",
                        "High": "2. high",
                        "Low": "3. low",
                        "Close": "4. close",
                        "Adj Close": "5. adjusted close",
                        "Volume": "6. volume"
                    })
                    df2.index = pd.to_datetime(df2.index)
                    df2 = df2.sort_index()
                    logger.info("yfinance success for %s (Ticker.history) rows=%d", sym, len(df2))
                    return df2
                # record attempt
                y_attempts.append((sym, "empty"))
            except Exception as e_y:
                logger.debug("yfinance attempt for %s failed: %s", sym, e_y)
                last_yf_error = str(e_y)
                y_attempts.append((sym, str(e_y)))

    else:
        logger.warning("yfinance not installed — cannot fallback to yfinance.")

    # --- 3) All attempts failed — produce helpful error showing AV message + yfinance attempts ---
    yf_summary = {"attempts": y_attempts, "last_error": last_yf_error}
    full_err = {
        "av_error": av_err_text,
        "yfinance_summary": yf_summary,
        "note": "Alpha Vantage was tried first (per your requirement). If AV returns Note about rate limits, check your API key or wait. If yfinance returns empty, try alternate ticker strings such as 'TCS' or 'TCS.NS'."
    }
    logger.error("Data fetch failed summary: %s", full_err)
    # Raise a single ValueError with details so the /predict endpoint returns them
    raise ValueError(json.dumps(full_err))

# ----------------------
# Bundle loader — robust for multiple shapes
# ----------------------
def load_bundle(models_dir: str = "models"):
    """
    Returns: scaler, features (list), lookback (int), model_kwargs (dict)
    Handles bundle.pkl being a dict with metadata or a bare scaler object.
    """
    p_bundle = os.path.join(models_dir, "bundle.pkl")
    p_scaler = os.path.join(models_dir, "scaler.pkl")
    if os.path.exists(p_bundle):
        raw = joblib.load(p_bundle)
        logger.debug("Loaded %s (type=%s)", p_bundle, type(raw))
    elif os.path.exists(p_scaler):
        raw = joblib.load(p_scaler)
        logger.debug("Loaded %s (type=%s)", p_scaler, type(raw))
    else:
        logger.warning("No bundle/scaler found in %s; using fallback defaults", models_dir)
        return None, FALLBACK_FEATURES, FALLBACK_LOOKBACK, FALLBACK_MODEL_KW

    if isinstance(raw, dict):
        scaler = raw.get("scaler", None)
        features = raw.get("features", FALLBACK_FEATURES)
        lookback = int(raw.get("lookback", FALLBACK_LOOKBACK))
        model_kwargs = raw.get("model_kwargs", FALLBACK_MODEL_KW)
        features = [str(f).strip().lower() for f in features]
        return scaler, features, lookback, model_kwargs

    if hasattr(raw, "transform"):
        # raw is a scaler instance (your current case)
        logger.info("bundle.pkl is a scaler instance; using fallback metadata")
        return raw, FALLBACK_FEATURES, FALLBACK_LOOKBACK, FALLBACK_MODEL_KW

    # list/tuple heuristics
    if isinstance(raw, (list, tuple)):
        scaler = None; features = None; lookback = FALLBACK_LOOKBACK; model_kwargs = FALLBACK_MODEL_KW
        for it in raw:
            if hasattr(it, "transform"):
                scaler = it
            if isinstance(it, (list, tuple)) and all(isinstance(x, (str, bytes)) for x in it):
                features = [str(x).strip().lower() for x in it]
            if isinstance(it, dict):
                if "lookback" in it:
                    lookback = it["lookback"]
                if "model_kwargs" in it:
                    model_kwargs = it["model_kwargs"]
        if scaler is not None and features is not None:
            return scaler, features, int(lookback), model_kwargs

    raise RuntimeError("Unsupported bundle.pkl format; inspect bundle with joblib.load manually.")

# ----------------------
# PyTorch model loader (if torch available)
# ----------------------
def load_pytorch_model(models_dir: str = "models"):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available in this environment.")

    model_pth = os.path.join(models_dir, "model.pth")
    if not os.path.exists(model_pth):
        raise FileNotFoundError("models/model.pth not found.")

    scaler, features, lookback, model_kwargs = load_bundle(models_dir=models_dir)
    if features is None:
        features = FALLBACK_FEATURES
    input_dim = int(model_kwargs.get("input_dim", len(features)))
    hidden_dim = int(model_kwargs.get("hidden_dim", FALLBACK_MODEL_KW["hidden_dim"]))
    num_layers = int(model_kwargs.get("num_layers", FALLBACK_MODEL_KW["num_layers"]))
    dropout = float(model_kwargs.get("dropout", FALLBACK_MODEL_KW["dropout"]))

    model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    state = torch.load(model_pth, map_location="cpu")

    if isinstance(state, dict):
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            try:
                model.load_state_dict(state)
            except Exception as e:
                found = False
                for v in state.values():
                    if isinstance(v, dict):
                        try:
                            model.load_state_dict(v)
                            found = True
                            break
                        except Exception:
                            continue
                if not found:
                    raise RuntimeError("Could not load state_dict from model.pth: " + str(e))
    elif isinstance(state, nn.Module):
        model = state
    elif isinstance(state, (list, tuple)) and isinstance(state[0], dict):
        model.load_state_dict(state[0])
    else:
        raise RuntimeError("Unexpected model.pth contents: " + str(type(state)))

    model.eval()
    return model, scaler, features, lookback

# ----------------------
# ONNX session loader
# ----------------------
_ONNX_SESSION = None
def get_onnx_session(path="models/model.onnx"):
    global _ONNX_SESSION
    if _ONNX_SESSION is None:
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not installed")
        if not os.path.exists(path):
            raise FileNotFoundError(path + " not found")
        _ONNX_SESSION = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _ONNX_SESSION

# ----------------------
# Unified predictor: AlphaVantage-first for fetching, robust scaler handling
# ----------------------
def predict_direction(stock_symbol: str, models_dir: str = "models"):
    """
    Returns {"symbol":..., "prob_up":float, "action":"buy"/"sell", "backend": "onnx"/"torch"}.
    """
    # load bundle (scaler/features/lookback/model_kwargs)
    scaler, features, lookback, model_kwargs = load_bundle(models_dir=models_dir)
    if features is None:
        features = FALLBACK_FEATURES

    # fetch data (AlphaVantage primary as requested)
    df = fetch_stock_data(stock_symbol)
    if df is None or df.empty:
        raise ValueError("No data for symbol: " + stock_symbol)

    # prepare features
    df_feat = prepare_features_for_bundle(df, features)
    if df_feat.shape[0] < lookback:
        raise ValueError(f"Not enough rows after feature engineering. Need >= {lookback}, got {df_feat.shape[0]}")

    # scale
    if scaler is None or not hasattr(scaler, "transform"):
        raise RuntimeError("Scaler missing or invalid in bundle (bundle.pkl appears not to contain a fitted scaler).")
    X_all = scaler.transform(df_feat.values)  # shape (n_rows, n_features)
    last_seq = X_all[-lookback:]  # (lookback, n_features)

    # ONNX path if available
    onnx_path = os.path.join(models_dir, "model.onnx")
    if ORT_AVAILABLE and os.path.exists(onnx_path):
        sess = get_onnx_session(onnx_path)
        X_input = last_seq[np.newaxis, :, :].astype(np.float32)
        ort_in = {sess.get_inputs()[0].name: X_input}
        outs = sess.run(None, ort_in)
        out_arr = np.array(outs[0]).ravel()
        prob = float(out_arr[0]) if out_arr.size >= 1 else float(np.squeeze(out_arr))
        action = "buy" if prob >= 0.5 else "sell"
        return {"symbol": stock_symbol, "prob_up": prob, "action": action, "backend":"onnx"}

    # PyTorch path if available
    model_pth = os.path.join(models_dir, "model.pth")
    if TORCH_AVAILABLE and os.path.exists(model_pth):
        model, _, _, _ = load_pytorch_model(models_dir=models_dir)
        X_tensor = torch.tensor(last_seq[np.newaxis, :, :], dtype=torch.float32)
        model.to(torch.device("cpu"))
        model.eval()
        with torch.no_grad():
            out = model(X_tensor)
        prob = float(out.cpu().numpy().ravel()[0])
        action = "buy" if prob >= 0.5 else "sell"
        return {"symbol": stock_symbol, "prob_up": prob, "action": action, "backend":"torch"}

    raise RuntimeError("No model available. Put models/model.onnx or models/model.pth + models/bundle.pkl (scaler) in place.")

# ----------------------
# Sentiment endpoint helper
# ----------------------
def load_sentiment_pipeline(candidates: List[str]):
    for p in candidates:
        if p and os.path.exists(p):
            logger.info("Loading sentiment pipeline from %s", p)
            return joblib.load(p)
    return None

# ----------------------
# Routes
# ----------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html = """
    <h1>Stock Direction API</h1>
    <p>Use /predict_direction for up/down probability (LSTM classifier).</p>
    <p>Use /debug_info to inspect environment & artifacts.</p>
    """
    return HTMLResponse(content=html)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/debug_info")
def debug_info():
    info = {
        "torch_available": TORCH_AVAILABLE,
        "onnxruntime_available": ORT_AVAILABLE,
        "yfinance_available": YFINANCE_AVAILABLE,
        "python_version": os.sys.version,
        "cwd": os.getcwd(),
        "artifacts": {
            "models/model.pth": os.path.exists("models/model.pth"),
            "models/model.onnx": os.path.exists("models/model.onnx"),
            "models/bundle.pkl": os.path.exists("models/bundle.pkl"),
            "models/scaler.pkl": os.path.exists("models/scaler.pkl")
        }
    }
    if os.path.exists("models/bundle.pkl"):
        try:
            raw = joblib.load("models/bundle.pkl")
            info["bundle_type"] = str(type(raw))
            if isinstance(raw, dict):
                info["bundle_keys"] = list(raw.keys())
                info["bundle_features_sample"] = [str(x) for x in raw.get("features", [])[:20]]
            elif hasattr(raw, "transform"):
                info["bundle_is_scaler"] = True
            else:
                info["bundle_repr"] = repr(raw)[:500]
        except Exception as e:
            info["bundle_load_error"] = str(e)
    return JSONResponse(content=info)

@app.get("/predict_direction")
def predict_direction_route(stock_symbol: str = Query("TCS.NS", description="Ticker to predict"),
                            models_dir: str = Query("models", description="models directory")):
    try:
        out = predict_direction(stock_symbol, models_dir=models_dir)
        return JSONResponse(content=out)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("predict_direction error: %s\n%s", e, tb)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict_sentiment")
def predict_sentiment(body: Dict = Body(...)):
    try:
        text = body.get("text", "")
        if not text:
            return JSONResponse(content={"error":"No text provided"}, status_code=400)
        candidates = [body.get("model_path")] if body.get("model_path") else []
        candidates += ["sentiment_model.pkl", os.path.join("models","sentiment_model.pkl")]
        pipeline = load_sentiment_pipeline(candidates)
        if pipeline is None:
            raise FileNotFoundError("Sentiment pipeline not found.")
        if hasattr(pipeline, "predict_proba"):
            prob = float(pipeline.predict_proba([text])[0,1])
            label = "positive" if prob >= 0.5 else "negative"
            return JSONResponse(content={"text": text, "prob_positive": prob, "label": label})
        else:
            pred = pipeline.predict([text])[0]
            return JSONResponse(content={"text": text, "label": str(pred)})
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("predict_sentiment error: %s\n%s", e, tb)
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting uvicorn on port %s", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
