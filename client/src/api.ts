import axios from "axios";
// const API_BASE = import.meta.env.VITE_BACKEND_URL as string;
const API_BASE = "http://127.0.0.1:8000";

export async function getPrediction(stock_symbol: string, days_ahead: number) {
  try {
    const res = await axios.get(`${API_BASE}/predict`, {
      params: { stock_symbol, days_ahead },
    });
    return res.data;
  } catch (err) {
    console.error("API error:", err);
    return null;
  }
}

export async function getData(stock_symbol: string) {
  try {
    const url = `${API_BASE}/fetch`;
    console.log("GET", url, { stock_symbol });
    const res = await axios.get(url, { params: { stock_symbol } });
    console.log("API response", res.status, res.data);
    if (res.status !== 200) throw new Error(`Status ${res.status}`);
    return res.data;
  } catch (err) {
    console.error("API error:", err);
    throw err; // rethrow so caller sees real error
  }
}
