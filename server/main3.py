# main.py
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os, pickle, traceback, logging, requests
from dotenv import load_dotenv
from typing import Optional

# Torch
import torch
import torch.nn as nn

# ===== Load Env & API Keys =====
load_dotenv()
API_KEY = os.getenv("MY_SECRET_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", API_KEY)
AV_BASE_URL = 'https://www.alphavantage.co/query'

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-predict")

# ===== FastAPI App =====
app = FastAPI(title="📈 Stock Price Prediction API",
              description="Predict stock prices using a trained BiLSTM model.",
              version="2.0.0")

origins = ["https://market-trend-analysis.vercel.app",
           "http://localhost:8080", "http://127.0.0.1:8080"]
app.add_middleware(CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ===== Model Setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Load artifacts once at startup
with open("bundle.pkl", "rb") as f:
    bundle = pickle.load(f)
scaler = bundle["scaler"]
lookback = bundle["lookback"]

model = BiLSTM()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# ===== Data Fetch =====
def fetch_stock_data(stock_symbol: str, period: str = "1y", timeout: int = 10) -> pd.DataFrame:
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

        if "Time Series (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df
        else:
            raise ValueError("Alpha Vantage returned error: " + str(data))

    except Exception as e:
        logger.warning("Alpha Vantage failed: %s", e)
        # fallback to yfinance
        import yfinance as yf
        df = yf.download(stock_symbol, period=period, progress=False)
        if df.empty:
            raise ValueError("yfinance returned empty DataFrame")
        df = df.rename(columns={"Close": "4. close"})
        return df

# ===== Prediction Function =====
def predict_stock_symbol(stock_symbol: str, days_ahead: int = 5):
    """
    Fetches stock data for a given symbol and predicts future prices using trained BiLSTM.

    Args:
        stock_symbol (str): Stock ticker symbol (e.g., 'TCS.NS', 'AAPL').
        days_ahead (int): Number of days to predict ahead.

    Returns:
        dict: { 'symbol': stock_symbol, 'predictions': [list of future prices] }
    """
    # 1. Fetch historical stock data
    df = fetch_stock_data(stock_symbol)
    if "4. close" not in df.columns:
        raise ValueError("Data missing '4. close' column in fetched data")

    close_prices = df["4. close"].dropna().values  # drop NaNs if any

    # 2. Ensure enough data for lookback
    if len(close_prices) < lookback:
        raise ValueError(f"Not enough data: got {len(close_prices)}, need {lookback}")

    # 3. Run prediction using trained model
    seq = scaler.transform(close_prices.reshape(-1, 1))[-lookback:]
    preds = []

    for _ in range(days_ahead):
        x = torch.tensor(seq.reshape(1, lookback, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_scaled = model(x).cpu().numpy().flatten()[0]
        pred = scaler.inverse_transform([[pred_scaled]])[0][0]
        preds.append(float(pred))

        # update sequence with new prediction
        seq = np.append(seq[1:], [[pred_scaled]], axis=0)

    return {"symbol": stock_symbol, "predictions": preds}


# ===== Routes =====
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>📈 Stock Price Prediction API (BiLSTM)</h1>
    <form action="/predict" method="get">
        <label>Stock Symbol:</label>
        <input name="stock_symbol" value="TCS.NS"><br><br>
        <label>Days Ahead:</label>
        <input name="days_ahead" type="number" value="5"><br><br>
        <button type="submit">Predict</button>
    </form>
    """

@app.get("/predict")
def predict(stock_symbol: str = Query("TCS.NS"),
            days_ahead: int = Query(5, ge=1, le=30),
            debug: Optional[bool] = False):
    try:
        df = fetch_stock_data(stock_symbol)
        if "4. close" not in df.columns:
            raise ValueError("Data missing '4. close' column")

        preds = predict_stock_prices(df["4. close"], days_ahead)

        return JSONResponse(content={
            "stock_symbol": stock_symbol,
            "days_ahead": days_ahead,
            "predictions": preds
        })
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Error: %s\n%s", exc, tb)
        if debug:
            return JSONResponse(content={"error": str(exc), "traceback": tb}, status_code=500)
        return JSONResponse(content={"error": str(exc)}, status_code=500)

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))