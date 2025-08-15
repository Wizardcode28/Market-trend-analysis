# main.py
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import requests
from typing import List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import logging
import traceback

load_dotenv()
API_KEY = os.getenv("MY_SECRET_API_KEY")
# ===== Try to import yfinance =====
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not installed. Only Alpha Vantage will be used.")

# ===== Alpha Vantage Configuration =====
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", API_KEY)
AV_BASE_URL = 'https://www.alphavantage.co/query'

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-predict")

# ===== FastAPI App =====
app = FastAPI(
    title="📈 Stock Price Prediction API",
    description="Predict stock prices using Random Forest regression with Alpha Vantage or Yahoo Finance data.",
    version="1.0.0"
)

# ===== CORS Middleware =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Helper Functions =====
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_stock_data(stock_symbol: str, period: str = "1y", timeout: int = 10) -> pd.DataFrame:
    """
    Fetch stock data from Alpha Vantage.
    If fails, fallback to yfinance if available.
    Raises ValueError with descriptive messages on failure.
    """
    logger.info("Fetching stock data for %s", stock_symbol)
    try:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": stock_symbol,
            "outputsize": "full",
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        response = requests.get(AV_BASE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("Alpha Vantage returned non-dict response")

        if "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            # columns in Alpha Vantage have names like "1. open", etc. We keep same names
            df = df.rename(columns={
                "1. open": "1. open",
                "2. high": "2. high",
                "3. low": "3. low",
                "4. close": "4. close",
                "5. adjusted close": "5. adjusted close",
                "6. volume": "6. volume"
            })
            # convert to numeric (float) where possible
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if df.empty:
                raise ValueError("Alpha Vantage returned empty DataFrame")
            return df
        else:
            # alpha vantage returned some error or reached limit
            err_text = data.get("Note") or data.get("Error Message") or "Alpha Vantage missing 'Time Series (Daily)'"
            raise ValueError(f"Alpha Vantage error: {err_text}")

    except Exception as e:
        logger.warning("Alpha Vantage failed for %s: %s", stock_symbol, e)
        # fallback to yfinance if available
        if YFINANCE_AVAILABLE:
            try:
                logger.info("Falling back to yfinance for %s", stock_symbol)
                df = yf.download(stock_symbol, period=period, progress=False)
                if df.empty:
                    raise ValueError("yfinance returned empty DataFrame")
                df = df.rename(columns={
                    "Open": "1. open",
                    "High": "2. high",
                    "Low": "3. low",
                    "Close": "4. close",
                    "Adj Close": "5. adjusted close",
                    "Volume": "6. volume"
                })
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df
            except Exception as e2:
                logger.error("yfinance fallback failed for %s: %s", stock_symbol, e2)
                raise ValueError(f"No data source available. AlphaVantage error: {e}. yfinance error: {e2}")
        else:
            raise ValueError(f"No data source available. AlphaVantage error: {e}. yfinance not installed.")


def predict_stock_prices(df: pd.DataFrame, days_ahead: int = 1):
    """
    Train a RandomForest on the fly and predict days_ahead.
    This function validates inputs and raises clear errors when data is insufficient.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame")

    df = df.copy()
    if '4. close' not in df.columns:
        raise ValueError("'4. close' (close price) column missing from dataframe")

    # feature engineering
    df['MA10'] = df['4. close'].rolling(window=10).mean()
    df['MA50'] = df['4. close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['4. close'])
    df['Lag1'] = df['4. close'].shift(1)
    df['Lag2'] = df['4. close'].shift(2)
    df = df.dropna()

    if df.shape[0] < 30:
        raise ValueError(f"Not enough data after feature engineering (rows={df.shape[0]}). Need more historical rows.")

    X = df[['MA10', 'MA50', 'RSI', 'Lag1', 'Lag2']]
    y = df['4. close']

    # train/test split (time-series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    if X_train.shape[0] < 10:
        raise ValueError("Not enough training rows after split to train model")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    try:
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        logger.error("Model training failed: %s", e)
        raise ValueError(f"Model training failed: {e}")

    # prepare last row for iterative prediction
    last_row = X.iloc[-1].copy()
    predictions = []
    confidences = []
    for _ in range(days_ahead):
        # ensure shape is correct
        try:
            pred_scaled = scaler.transform([last_row])
            all_tree_preds = np.array([tree.predict(pred_scaled)[0] for tree in model.estimators_])
            pred_price = model.predict(pred_scaled)[0]
            # Confidence: higher if trees agree
            confidence = 100 * (1 - np.std(all_tree_preds)/pred_price)
            confidence = max(0, min(100, confidence))  # clamp to 0-100
            confidence = round(confidence, 2) 
        except Exception as e:
            logger.error("Prediction step failed: %s", e)
            raise ValueError(f"Prediction step failed: {e}")

        predictions.append(float(round(pred_price, 4)))
        confidences.append(confidence)

        # update last_row features for next step
        # careful with indexing — explicitly set values
        prev_lag1 = float(last_row['Lag1'])
        last_row['Lag2'] = prev_lag1
        last_row['Lag1'] = float(pred_price)
        # update moving averages approx
        last_row['MA10'] = float((last_row['MA10'] * 10 - prev_lag1 + pred_price) / 10)
        last_row['MA50'] = float((last_row['MA50'] * 50 - prev_lag1 + pred_price) / 50)
        # RSI recompute using last 3 values (approx)
        try:
            last_row['RSI'] = float(calculate_rsi(pd.Series([last_row['Lag2'], last_row['Lag1'], pred_price])).iloc[-1])
        except Exception:
            # fallback: keep previous RSI
            last_row['RSI'] = float(last_row.get('RSI', 50.0))

    mae = None
    r2 = None
    try:
        y_test_pred = model.predict(X_test_scaled)
        mae = float(mean_absolute_error(y_test, y_test_pred))
        r2 = float(r2_score(y_test, y_test_pred))
    except Exception as e:
        logger.warning("Failed to compute metrics: %s", e)

    return predictions,confidences, mae, r2


# ===== FastAPI Routes =====
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <h1>📈 Stock Price Prediction API</h1>
    <form action="/predict" method="get">
        <label>Stock Symbol:</label>
        <input name="stock_symbol" value="TCS.NS"><br><br>
        <label>Days Ahead:</label>
        <input name="days_ahead" type="number" value="5"><br><br>
        <button type="submit">Predict</button>
    </form>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(
    stock_symbol: str = Query("TCS.NS", description="Stock ticker symbol"),
    days_ahead: int = Query(5, ge=1, le=30, description="Days ahead to predict"),
    debug: Optional[bool] = Query(False, description="If true, return traceback in response (for debugging only)")
):
    logger.info("Received predict request: symbol=%s days_ahead=%s debug=%s", stock_symbol, days_ahead, debug)
    try:
        # Basic validation
        if not isinstance(stock_symbol, str) or not stock_symbol.strip():
            raise ValueError("stock_symbol must be a non-empty string")
        if not isinstance(days_ahead, int) or days_ahead < 1 or days_ahead > 30:
            raise ValueError("days_ahead must be an integer between 1 and 30")

        # Fetch data (this may raise a ValueError with clear message)
        df = fetch_stock_data(stock_symbol)

        # Validate df
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("No data returned for symbol: " + stock_symbol)

        # Predict
        preds,confs, mae, r2 = predict_stock_prices(df, days_ahead)

        result = {
            "stock_symbol": stock_symbol,
            "days_ahead": days_ahead,
            "predictions": preds,
            "confidence": confs,  # send as a list
            "mae": mae,
            "r2": r2
        }
        logger.info("Prediction success: %s", result)
        return JSONResponse(content=result)
    except Exception as exc:
        # print traceback to server console
        tb = traceback.format_exc()
        logger.error("Error in /predict: %s\n%s", exc, tb)
        # If debug requested, return traceback in response to help debugging
        if debug:
            return JSONResponse(content={"error": str(exc), "traceback": tb}, status_code=500)
        # Otherwise return concise error
        return JSONResponse(content={"error": str(exc)}, status_code=500)

# ===== Run with: uvicorn main:app --reload =====
