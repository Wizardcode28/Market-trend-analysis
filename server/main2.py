from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import requests
from typing import Optional
import logging
import traceback
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.callbacks import EarlyStopping
import pickle

try:
    import torch
    TORCH_AVAILABLE=True
except Exception:
    TORCH_AVAILABLE=False
load_dotenv()

API_KEY = os.getenv("MY_SECRET_API_KEY")
# ===== Try to import yfinance =====
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not installed. Will rely on Alpha Vantage and CSV fallback.")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", API_KEY)
AV_BASE_URL = "https://www.alphavantage.co/query"

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-predict")

# ===== FastAPI App =====
app = FastAPI(
    title="📈 Stock Price Prediction API (BiLSTM)",
    description="Predict stock prices using a stacked BiLSTM with smart datasource selection (Yahoo Finance for India, Alpha Vantage for US; CSV fallback).",
    version="2.0.0",
)
origins = [
    "https://market-trend-analysis.vercel.app",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # tighten in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================
# Utils: Indicators (no 'ta' dependency)
# ============================================
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # expects columns: open, high, low, close, volume
    out = df.copy()
    out["rsi"]  = calculate_rsi(out["close"], 14)
    out["ma20"] = out["close"].rolling(20).mean()
    out["ma50"] = out["close"].rolling(50).mean()
    out = out.dropna()
    return out

# ============================================
# Datasource: Alpha Vantage and Yahoo choice
# ============================================
def is_indian_symbol(symbol: str) -> bool:
    s = symbol.upper()
    return s.endswith(".NS") or s.endswith(".BO")

def fetch_from_alphavantage(symbol: str, timeout: int = 12) -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    resp = requests.get(AV_BASE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "Time Series (Daily)" not in data:
        msg = data.get("Note") or data.get("Error Message") or "Alpha Vantage response missing time series."
        raise ValueError(msg)
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Normalize columns to simple names
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. adjusted close": "adj_close",
        "6. volume": "volume"
    })
    keep = ["open", "high", "low", "close", "volume"]
    df = df[keep]
    return df

def fetch_from_yfinance(symbol: str, period: str = "5y") -> pd.DataFrame:
    if not YFINANCE_AVAILABLE:
        raise ValueError("yfinance not installed")
    df = yf.download(symbol, period=period, progress=False)
    if df.empty:
        raise ValueError("yfinance returned empty data")
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    keep = ["open", "high", "low", "close", "volume"]
    df = df[keep]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

logger = logging.getLogger(__name__)

def fetch_stock_data(symbol: str) -> pd.DataFrame:
    last_exception = None

    # 1️⃣ Try Yahoo Finance
    try:
        import yfinance as yf
        df = yf.download(symbol, period="6mo", interval="1d")
        if not df.empty:
            df.reset_index(inplace=True)
            df = df.rename(columns=str.lower).set_index("date")
            df = df[["open", "high", "low", "close", "volume"]]
            logger.info("Fetched %s data from Yahoo Finance", symbol)
            return df
    except Exception as e:
        logger.warning("Yahoo fetch failed for %s: %s", symbol, e)
        last_exception = e

    # 2️⃣ Try AlphaVantage
    try:
        df = fetch_from_alphavantage(symbol)  # assumes this function exists
        logger.info("Fetched %s data from AlphaVantage", symbol)
        return df
    except Exception as e:
        logger.warning("AlphaVantage fetch failed for %s: %s", symbol, e)
        last_exception = e

    # 3️⃣ Try CSV fallback
    csv_path = "infolimpioavanzadoTarget.csv"
    if os.path.exists(csv_path):
        try:
            df_csv = pd.read_csv(csv_path)
            df_csv.columns = [c.strip().lower() for c in df_csv.columns]
            df_csv = df_csv[df_csv["ticker"].str.upper() == symbol.upper()].copy()
            df_csv["date"] = pd.to_datetime(df_csv["date"])
            df_csv = df_csv.sort_values("date").set_index("date")
            df_csv = df_csv[["open", "high", "low", "close", "volume"]]
            logger.info("Using CSV fallback for %s", symbol)
            return df_csv
        except Exception as e:
            last_exception = e

    # 4️⃣ Try bundle.pkl fallback
    bundle_path = "/mnt/data/bundle.pkl"
    if os.path.exists(bundle_path):
        try:
            with open(bundle_path, "rb") as f:
                bundle = pickle.load(f)
            # Assume bundle is a dict with symbols as keys and DataFrames as values
            if symbol.upper() in bundle:
                df_bundle = bundle[symbol.upper()]
                logger.info("Using bundle.pkl fallback for %s", symbol)
                return df_bundle
        except Exception as e:
            last_exception = e

    # 5️⃣ Optional: Try model.pth (if it contains data, not weights)
    model_path = "/mnt/data/model.pth"
    if os.path.exists(model_path):
        try:
            model_data = torch.load(model_path, map_location="cpu")
            # If model_data contains a DataFrame or dict of DataFrames keyed by symbols
            if isinstance(model_data, dict) and symbol.upper() in model_data:
                df_model = model_data[symbol.upper()]
                logger.info("Using model.pth fallback for %s", symbol)
                return df_model
        except Exception as e:
            last_exception = e

    # ❌ If all fail
    raise ValueError(f"All sources failed for {symbol}. Last exception: {last_exception}")
# ============================================
# Sequences & Model
# ============================================
def make_sequences(scaled_array: np.ndarray, close_idx: int, lookback: int, horizon: int):
    X, y = [], []
    for i in range(lookback, len(scaled_array) - horizon):
        X.append(scaled_array[i - lookback:i])
        y.append(scaled_array[i:i + horizon, close_idx])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y

def build_bilstm(input_dim: int, lookback: int, horizon: int) -> tf.keras.Model:
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, input_dim)),
        Dropout(0.3),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.3),
        Dense(horizon)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_or_load_model(symbol: str, df_raw: pd.DataFrame, lookback=120, horizon=30, force_retrain=False):
    """
    Returns (model, scaler, meta_dict, metrics_dict)
    Saves/loads from models_tf/{SYMBOL}_bilstm.h5 + _scaler.pkl
    """
    os.makedirs("models_tf", exist_ok=True)
    model_path = f"models_tf/{symbol}_bilstm.h5"
    scaler_path = f"models_tf/{symbol}_scaler.pkl"

    # Prepare data
    df = add_features(df_raw)  # adds rsi, ma20, ma50
    if len(df) < lookback + horizon + 50:
        raise ValueError(f"Not enough rows after feature engineering (got {len(df)}).")

    # Scale ALL columns used by sequences
    cols = ["open", "high", "low", "close", "volume", "rsi", "ma20", "ma50"]
    df_w = df[cols].copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_w.values)
    close_idx = cols.index("close")

    X, y = make_sequences(scaled, close_idx, lookback, horizon)
    if len(X) == 0:
        raise ValueError("Not enough sequences created; try smaller lookback/horizon")

    # time-based split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Try load
    if os.path.exists(model_path) and os.path.exists(scaler_path) and not force_retrain:
        try:
            model = load_model(model_path)
            meta = {"cols": cols, "close_idx": close_idx, "lookback": lookback, "horizon": horizon}
            saved_scaler = joblib.load(scaler_path)
            # metrics from quick validation
            val_pred = model.predict(X_val, verbose=0)
            # inverse transform first horizon day to compute metrics
            mae, r2, rmse = _first_horizon_metrics(y_val, val_pred, scaler, close_idx, scaled.shape[1])
            metrics = {"mae": mae, "r2": r2, "rmse": rmse}
            return model, saved_scaler, meta, metrics
        except Exception as e:
            logger.warning("Failed to load saved model, retraining: %s", e)

    # Train fresh
    model = build_bilstm(input_dim=X.shape[2], lookback=lookback, horizon=horizon)
    early = EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early],
        verbose=0
    )

    # Save artifacts
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(cols, f"{scaler_path}.cols")

    # Compute metrics on validation (first horizon day)
    val_pred = model.predict(X_val, verbose=0)
    mae, r2, rmse = _first_horizon_metrics(y_val, val_pred, scaler, close_idx, scaled.shape[1])
    metrics = {"mae": mae, "r2": r2, "rmse": rmse}

    return model, scaler, {"cols": cols, "close_idx": close_idx, "lookback": lookback, "horizon": horizon}, metrics

def _first_horizon_metrics(y_true_scaled, y_pred_scaled, scaler, close_idx, n_features):
    # inverse-transform only the close column safely
    true_1d, pred_1d = [], []
    for i in range(len(y_pred_scaled)):
        temp = np.zeros((y_pred_scaled.shape[1], n_features))
        temp[:, close_idx] = y_true_scaled[i]
        inv_true = scaler.inverse_transform(temp)[:, close_idx]

        temp[:, close_idx] = y_pred_scaled[i]
        inv_pred = scaler.inverse_transform(temp)[:, close_idx]

        true_1d.append(inv_true[0])
        pred_1d.append(inv_pred[0])

    true_1d = np.array(true_1d)
    pred_1d = np.array(pred_1d)
    mae = float(mean_absolute_error(true_1d, pred_1d))
    rmse = float(np.sqrt(mean_squared_error(true_1d, pred_1d)))
    mean_y = float(np.mean(true_1d))
    r2 = float(r2_score(true_1d, pred_1d)) if len(true_1d) > 1 else None
    return mae, r2, rmse

def forecast_next_days(model, df_raw: pd.DataFrame, scaler: MinMaxScaler, meta: dict, days_ahead: int):
    cols = meta["cols"]
    close_idx = meta["close_idx"]
    lookback = meta["lookback"]
    horizon = meta["horizon"]

    df = add_features(df_raw)
    df_w = df[cols].copy()
    scaled = scaler.transform(df_w.values)

    last_seq = scaled[-lookback:]
    last_seq = np.expand_dims(last_seq, axis=0)  # (1, lookback, n_features)

    # Model always outputs 'horizon' steps; if user asks fewer, slice
    pred_scaled = model.predict(last_seq, verbose=0)[0]  # (horizon,)
    if days_ahead < horizon:
        pred_scaled = pred_scaled[:days_ahead]

    # Inverse transform just for close
    temp = np.zeros((len(pred_scaled), len(cols)))
    temp[:, close_idx] = pred_scaled
    inv = scaler.inverse_transform(temp)[:, close_idx]
    return inv.tolist()

def confidence_from_rmse(metrics: dict, df_raw: pd.DataFrame) -> float:
    """Crude confidence heuristic: higher if RMSE is small relative to mean close."""
    try:
        mean_close = float(df_raw["close"].tail(200).mean())
        rmse = metrics.get("rmse", None)
        if rmse is None or mean_close <= 0:
            return 50.0
        c = 100.0 * (1.0 - (rmse / mean_close))
        return float(np.clip(c, 0.0, 100.0))
    except Exception:
        return 50.0

# ============================================
# Routes
# ============================================
@app.get("/", response_class=HTMLResponse)
@app.head("/")
def home():
    html = """
    <h1>📈 Stock Price Prediction API (BiLSTM)</h1>
    <form action="/predict" method="get">
        <label>Stock Symbol:</label>
        <input name="stock_symbol" value="TCS.NS"><br><br>
        <label>Days Ahead (1-30):</label>
        <input name="days_ahead" type="number" value="5" min="1" max="30"><br><br>
        <button type="submit">Predict</button>
    </form>
    """
    return HTMLResponse(content=html)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(
    stock_symbol: str = Query("TCS.NS", description="Stock ticker symbol"),
    days_ahead: int = Query(5, ge=1, le=30, description="Days ahead to predict"),
    force_retrain: Optional[bool] = Query(False, description="Force retrain even if cached model exists"),
    debug: Optional[bool] = Query(False, description="Include traceback in response for debugging")
):
    logger.info("Predict request: %s, days_ahead=%d, force_retrain=%s", stock_symbol, days_ahead, force_retrain)
    try:
        if not stock_symbol or not isinstance(stock_symbol, str):
            raise ValueError("stock_symbol must be a non-empty string")

        # 1) Fetch data (smart chooser → CSV fallback)
        df = fetch_stock_data(stock_symbol)
        if df is None or df.empty:
            raise ValueError(f"No data returned for {stock_symbol}")

        # 2) Train or load model
        model, scaler, meta, metrics = train_or_load_model(
            symbol=stock_symbol,
            df_raw=df,
            lookback=min(120,len(df)/2),
            horizon=30,
            force_retrain=bool(force_retrain)
        )

        # 3) Forecast
        preds = forecast_next_days(model, df_raw=df, scaler=scaler, meta=meta, days_ahead=days_ahead)

        # 4) Confidence
        conf = confidence_from_rmse(metrics, df)
        conf_list = [round(conf, 2)] * len(preds)

        result = {
            "stock_symbol": stock_symbol,
            "days_ahead": days_ahead,
            "predictions": [float(round(x, 4)) for x in preds],
            "confidence": conf_list,
            "mae": metrics.get("mae"),
            "r2": metrics.get("r2"),
            "rmse": metrics.get("rmse"),
            "model_files": {
                "weights_h5": f"models_tf/{stock_symbol}_bilstm.h5",
                "scaler_pkl": f"models_tf/{stock_symbol}_scaler.pkl"
            }
        }
        return JSONResponse(content=result)

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Error: %s\n%s", exc, tb)
        if debug:
            return JSONResponse(status_code=500, content={"error": str(exc), "traceback": tb})
        return JSONResponse(status_code=500, content={"error": str(exc)})

# ===== Run with: uvicorn main:app --reload =====
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)