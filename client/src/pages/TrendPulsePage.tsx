import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState,useEffect } from "react";
// import { getData } from "../api";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
type PriceRow = { date: string; close: number | null; ma20?: number | null; ma50?: number | null };
type VolumeRow = { date: string; volume: number };
type RsiRow = { date: string; rsi: number | null };
const tickers = [
  { symbol: "AAPL",       name: "Apple Inc." },
  { symbol: "AMZN",       name: "Amazon.com, Inc." },
  { symbol: "AMD",        name: "Advanced Micro Devices, Inc." },
  { symbol: "GOOGL",      name: "Alphabet Inc. (Class A)" },
  { symbol: "MSFT",       name: "Microsoft Corporation" },
  { symbol: "TSLA",       name: "Tesla, Inc." },
  { symbol: "META",       name: "Meta Platforms, Inc." },
  { symbol: "NFLX",       name: "Netflix, Inc." },
  { symbol: "NVDA",       name: "NVIDIA Corporation" },
  { symbol: "INTC",       name: "Intel Corporation" },
  { symbol: "INFY.NS",    name: "Infosys Limited" },
  { symbol: "HDFCBANK.NS",name: "HDFC Bank Limited" },
  { symbol: "RELIANCE.NS", name: "Reliance Industries Limited" },
  { symbol: "TCS.NS",     name: "Tata Consultancy Services Limited" }
];


export default function TrendPulsePage() {

const [loading, setLoading] = useState(false);
const [error, setError] = useState<string | null>(null);
const [price_data, setprice_data] = useState([]);
const [volume_data, setvolume_data] = useState([]);
const [rsi_data, setrsi_data] = useState([]);
const [symbol, setSymbol] = useState<string>("AAPL"); // default

    async function fetchData(stockSymbol: string) {
      setLoading(true);
      setError(null);
      try{
        const result= await getData(stockSymbol);
        console.log("API Result:", result);
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
        if (!Array.isArray(result.price_data) || result.price_data.length === 0) {
        setError("No Data returned by API");
        setLoading(false);
        return;
        }
        setprice_data(result.price_data as PriceRow[])
        setvolume_data((result.volume_data ?? []) as VolumeRow[]);
        setrsi_data((result.rsi_data ?? []) as RsiRow[]);
      }
      catch (err: any) {
      console.error("Fetch error:", err);
      // setError(err?.message ? String(err.message) : "Unknown error");
      throw err;
    } finally {
      setLoading(false);
    }
    }

    useEffect(() => {
      fetchData(symbol);
    }, [symbol]);

  // Defensive: show placeholder if chart data is empty
  const hasPrice = price_data && price_data.length > 0;
  const hasVolume = volume_data && volume_data.length > 0;
  const hasRsi = rsi_data && rsi_data.length > 0;
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient mb-2">TrendPulse™ Meter</h1>
          <p className="text-muted-foreground">Real-time market sentiment analysis with advanced charts</p>
        </div>
        {/* Controls: symbol input, fetch button */}
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

          <Button onClick={() => fetchData(symbol)} disabled={loading}>
            {loading ? "Loading..." : "Get Data"}
          </Button>

          <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse-glow mr-2" />
            Live Data
          </Badge>
        </div>

      </div>

      {/* TrendPulse Meter */}
      <Card className="glass-effect p-6 border-border/50">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-foreground">TrendPulse™ Meter</h2>
            <p className="text-muted-foreground">Real-time market sentiment analysis with advanced charts</p>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
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

        {/* Advanced Charts */}
        <Tabs defaultValue="price" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="price">Price & Moving Averages</TabsTrigger>
            <TabsTrigger value="volume">Trading Volume</TabsTrigger>
            <TabsTrigger value="rsi">RSI Analysis</TabsTrigger>
          </TabsList>
          
          <TabsContent value="price" className="mt-6">
            <div className="h-80 w-full">
              <h3 className="font-semibold text-foreground mb-4">{symbol} Stock Price with Moving Averages</h3>
              {!hasPrice ? (
                <div className="text-center text-muted-foreground">No price data available.</div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={price_data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" />
                    <YAxis stroke="hsl(var(--muted-foreground))" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                      }}
                    />
                    <Line type="monotone" dataKey="close" stroke="hsl(var(--primary))" strokeWidth={2} name="Close Price" />
                    <Line type="monotone" dataKey="ma20" stroke="hsl(var(--accent))" strokeWidth={2} strokeDasharray="5 5" name="20-Day MA" />
                    <Line type="monotone" dataKey="ma50" stroke="hsl(var(--muted-foreground))" strokeWidth={2} strokeDasharray="10 5" name="50-Day MA" />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="volume" className="mt-6">
            <div className="h-80 w-full">
              <h3 className="font-semibold text-foreground mb-4">{symbol} Trading Volume</h3>
{!hasVolume ? (
                <div className="text-center text-muted-foreground">No volume data available.</div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={volume_data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" />
                    <YAxis stroke="hsl(var(--muted-foreground))" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "8px",
                      }}
                    />
                    <Bar dataKey="volume" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="rsi" className="mt-6">
            <div className="h-80 w-full">
              <h3 className="font-semibold text-foreground mb-4">{symbol} RSI (Relative Strength Index)</h3>
              {!hasRsi ? (
                <div className="text-center text-muted-foreground">No RSI data available.</div>
              ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rsi_data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" />
                  <YAxis domain={[0, 100]} stroke="hsl(var(--muted-foreground))" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px"
                    }} 
                  />
                  <ReferenceLine y={70} stroke="hsl(var(--destructive))" strokeDasharray="5 5" label="Overbought (70)" />
                  <ReferenceLine y={30} stroke="hsl(var(--primary))" strokeDasharray="5 5" label="Oversold (30)" />
                  <Line type="monotone" dataKey="rsi" stroke="hsl(var(--accent))" strokeWidth={3} name="RSI" />
                </LineChart>
              </ResponsiveContainer>
             )}
            </div>
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  );
}