import React, { useEffect, useState } from "react";
import { getPrediction } from "../api";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  ArrowUpRight,
  ArrowDownRight,
  Brain,
  Zap,
  DollarSign,
  Bitcoin,
  BarChart3
} from "lucide-react";

const tickers = [
  { symbol: "AAPL", name: "Apple Inc." },
  { symbol: "TSLA", name: "Tesla, Inc." },
  { symbol: "TCS.NS", name: "Tata Consultancy Services" },
  { symbol: "GOOGL", name: "Alphabet Inc." },
];

export default function DashboardPage() {
  // UI state
  const [selectedTicker, setSelectedTicker] = useState("AAPL");
  const [marketData, setMarketData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Controls for user input
  const [symbol, setSymbol] = useState<string>("AAPL"); // default
  const [days, setDays] = useState<number>(5);

  const aiInsights = [
    { title: "Market Volatility Alert", description: "Increased volatility detected in crypto markets due to regulatory news.", severity: "warning" },
    { title: "Tech Stocks Momentum", description: "AI predicts continued upward momentum in major tech stocks for next 48 hours.", severity: "positive" },
    { title: "Gold Support Level", description: "Gold approaching key support level at $2,030. Watch for potential reversal.", severity: "info" },
  ];

  // Fetch function (callable from mount or button)
  async function fetchPredictions(stockSymbol: string, daysAhead: number) {
    setLoading(true);
    setError(null);
    setMarketData([]);
    try {
      const result = await getPrediction(stockSymbol, daysAhead);

      if (!result) {
        setError("No response from API");
        setLoading(false);
        return;
      }

      if (result.error) {
        // backend returned error object
        setError(String(result.error));
        setLoading(false);
        return;
      }

      if (!Array.isArray(result.predictions) || result.predictions.length === 0) {
        setError("No predictions returned by API");
        setLoading(false);
        return;
      }

      // result.confidence may be an array (per-day) or single value
      const confArray = Array.isArray(result.confidence) ? result.confidence : null;
      const confSingle = !Array.isArray(result.confidence) && result.confidence !== undefined ? Number(result.confidence) : null;

      const predictions = result.predictions.map((predRaw: any, idx: number) => {
        const pred = Number(predRaw);
        // get per-day confidence if available, otherwise use single value or fallback 0
        const rawConf = confArray ? confArray[idx] : confSingle;
        const confNum = rawConf === undefined || rawConf === null ? 0 : Number(rawConf);
        const confRounded = Number.isFinite(confNum) ? Number(confNum.toFixed(2)) : 0;

        return {
          symbol: result.stock_symbol ?? stockSymbol,
          // display predicted price nicely
          price: Number.isFinite(pred) ? `$${pred.toFixed(2)}` : "N/A",
          // difference from previous predicted day (for days > 0)
          change:
            idx === 0
              ? "N/A"
              : (pred - Number(result.predictions[idx - 1]) >= 0 ? "+" : "-") +
                Math.abs(pred - Number(result.predictions[idx - 1])).toFixed(2),
          prediction: "Predicted Price",
          confidence: confRounded, // number 0-100
          trend:
            idx === 0
              ? "up"
              : pred >= Number(result.predictions[idx - 1])
              ? "up"
              : "down",
        };
      });

      setMarketData(predictions);
    } catch (err: any) {
      console.error("Fetch error:", err);
      setError(err?.message ? String(err.message) : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  // fetch default symbol once on mount
  useEffect(() => {
    fetchPredictions(symbol, days);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="p-6 space-y-6">
      {/* Header + Controls */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient mb-2">Market Dashboard</h1>
          <p className="text-muted-foreground">AI-powered market analysis and predictions</p>
        </div>

        {/* Controls: symbol input, days, fetch button */}
        <div className="flex items-center gap-2">
          <select
            className="border rounded px-3 py-1 bg-card/80"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          >
            {tickers.map((t) => (
              <option key={t.symbol} value={t.symbol}>
                {t.name} ({t.symbol})
              </option>
            ))}
          </select>

          <input
            className="w-20 border rounded px-3 py-1 bg-card/80"
            type="number"
            value={days}
            min={1}
            max={30}
            onChange={(e) => setDays(Math.max(1, Math.min(30, Number(e.target.value || 1))))}
            aria-label="days ahead"
          />
          <Button onClick={() => fetchPredictions(symbol, days)} disabled={loading}>
            {loading ? "Loading..." : "Get Predictions"}
          </Button>

          <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse-glow mr-2" />
            Live Data
          </Badge>
        </div>
      </div>

      {/* TrendPulse Meter (unchanged) */}
      <Card className="glass-effect p-6 border-border/50">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-foreground">TrendPulse™ Meter</h2>
            <p className="text-muted-foreground">Real-time market sentiment analysis</p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-primary mb-1">74%</div>
            <div className="text-sm text-muted-foreground">Overall Bullish</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-accent mb-1">86</div>
            <div className="text-sm text-muted-foreground">Fear & Greed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-primary mb-1">+12%</div>
            <div className="text-sm text-muted-foreground">Volume Surge</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-accent mb-1">92%</div>
            <div className="text-sm text-muted-foreground">AI Confidence</div>
          </div>
        </div>
      </Card>

      {/* Market Predictions */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card className="glass-effect p-6 border-border/50">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-foreground">AI Market Predictions</h2>
              <Button variant="outline" size="sm" onClick={() => fetchPredictions(symbol, days)} disabled={loading}>
                <Brain className="w-4 h-4 mr-2" />
                Update Predictions
              </Button>
            </div>

            <div className="space-y-4">
              {error && <p className="text-destructive">Error: {error}</p>}

              {loading ? (
                <p>Loading market data...</p>
              ) : marketData.length > 0 ? (
                marketData.map((item, index) => (
                  <div key={index} className="flex items-center justify-between p-4 rounded-lg bg-card/50 border border-border/30">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 rounded-full bg-gradient-primary flex items-center justify-center">
                        {item.symbol.includes("BTC") ? <Bitcoin className="w-5 h-5 text-white" /> :
                         item.symbol.includes("USD") ? <DollarSign className="w-5 h-5 text-white" /> :
                         <BarChart3 className="w-5 h-5 text-white" />}
                      </div>
                      <div>
                        <div className="font-semibold text-foreground">{item.symbol}</div>
                        <div className="text-sm text-muted-foreground">{item.price}</div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <div className={`flex items-center gap-1 ${item.trend === "up" ? "text-primary" : "text-destructive"}`}>
                        {item.trend === "up" ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
                        <span className="font-medium">{item.change}</span>
                      </div>

                      <div className="text-right min-w-[100px]">
                        <div className="text-sm font-medium text-foreground">{item.prediction}</div>
                        <div className="flex items-center gap-2 mt-1">
                          {/* Progress expects a numeric 0-100 */}
                          <Progress value={Number(item.confidence)} className="w-16 h-2" />
                          <span className="text-xs text-muted-foreground">{Number(item.confidence).toFixed(2)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <p>No market data available.</p>
              )}
            </div>
          </Card>
        </div>

        {/* AI Insights */}
        <div>
          <Card className="glass-effect p-6 border-border/50">
            <h2 className="text-xl font-semibold text-foreground mb-6">AI Insights</h2>
            <div className="space-y-4">
              {aiInsights.map((insight, index) => (
                <div key={index} className="p-4 rounded-lg bg-card/30 border border-border/20">
                  <div className="flex items-start gap-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      insight.severity === 'warning' ? 'bg-destructive/20 text-destructive' :
                      insight.severity === 'positive' ? 'bg-primary/20 text-primary' :
                      'bg-accent/20 text-accent'
                    }`}>
                      <Brain className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="font-medium text-foreground mb-1">{insight.title}</h3>
                      <p className="text-sm text-muted-foreground">{insight.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <Button className="w-full mt-6 bg-gradient-primary hover:opacity-90 text-white">
              View All Insights
            </Button>
          </Card>
        </div>
      </div>
    </div>
  );
}
