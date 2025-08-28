import React, { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { getSentiment } from "../api";
import {
  TrendingUp,
  Clock,
  ExternalLink,
  Brain,
  AlertTriangle,
} from "lucide-react";

type SentimentLabel = "positive" | "neutral" | "negative";
type ImpactLabel = "high" | "medium" | "low";

interface NewsItem {
  title: string;
  summary: string;
  source: string;
  time: string; // e.g. "2 hours ago"
  sentiment: SentimentLabel;
  impact: ImpactLabel;
  aiConfidence: number | 98; // 0..100
  // optional: url?: string
}

export default function NewsPage() {
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    async function fetchData() {
      const data = await getSentiment("AAPL", 5);
      if (data) setNewsItems(data);
    }
    fetchData();
  }, []);
  // You can change this default company or make it a controlled input
  const companyQuery = "Apple";
  const numArticles = 6;

  useEffect(() => {
    let isMounted = true;

    async function loadNews() {
      setLoading(true);
      setError(null);

      try {
        // adjust URL to your backend address if necessary (e.g., proxy or full URL)
        const resp = await fetch(
          `/sentiment?company=${encodeURIComponent(companyQuery)}&num_articles=${numArticles}`
        );
        if (!resp.ok) {
          const text = await resp.text();
          throw new Error(`Server error ${resp.status}: ${text}`);
        }

        const json = await resp.json();

        // backend returns { newsItems: [...] } per server implementation
        const items: any[] = json.newsItems ?? json.news_items ?? json.items ?? [];

        // normalize incoming items defensively
        const normalized: NewsItem[] = items.map((it) => ({
          title: (it.title ?? it.headline ?? "").toString(),
          summary: (it.summary ?? it.description ?? it.snippet ?? "").toString(),
          source: (it.source ?? it.source_name ?? "Unknown").toString(),
          time: (it.time ?? it.published_at ?? it.publishedAt ?? "unknown").toString(),
          sentiment: (it.sentiment ?? "neutral") as SentimentLabel,
          impact: (it.impact ?? "low") as ImpactLabel,
          aiConfidence: Number(it.aiConfidence ?? it.confidence ?? it.score ?? 0),
        }));

        if (isMounted) {
          setNewsItems(normalized);
        }
      } catch (err: any) {
        console.error("Failed to load news:", err);
        if (isMounted) setError(err?.message ?? String(err));
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    loadNews();

    return () => {
      isMounted = false;
    };
  }, [companyQuery, numArticles]);

  const getSentimentColor = (sentiment: SentimentLabel) => {
    switch (sentiment) {
      case "positive":
        return "text-primary border-primary/30 bg-primary/5";
      case "negative":
        return "text-destructive border-destructive/30 bg-destructive/5";
      default:
        return "text-accent border-accent/30 bg-accent/5";
    }
  };

  const getImpactColor = (impact: ImpactLabel) => {
    switch (impact) {
      case "high":
        return "text-destructive";
      case "medium":
        return "text-accent";
      default:
        return "text-muted-foreground";
    }
  };

  // UI helpers
  if (loading) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gradient mb-2">News & Insights</h1>
            <p className="text-muted-foreground">AI-powered market news analysis</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5">
              <Brain className="w-4 h-4 mr-2" />
              Loading...
            </Badge>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {/* lightweight skeletons */}
          {Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="glass-effect p-6 border-border/50 animate-pulse">
              <div className="h-6 bg-muted-foreground/10 rounded w-1/2 mb-4" />
              <div className="h-4 bg-muted-foreground/10 rounded w-3/4 mb-2" />
              <div className="h-3 bg-muted-foreground/10 rounded w-2/3 mb-4" />
              <div className="flex justify-between items-center">
                <div className="h-6 bg-muted-foreground/10 rounded w-24" />
                <div className="h-8 bg-muted-foreground/10 rounded w-28" />
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient mb-2">News & Insights</h1>
          <p className="text-muted-foreground">AI-powered market news analysis and sentiment tracking</p>
          <p className="text-sm text-muted-foreground mt-1">
            Showing results for <strong>{companyQuery}</strong>
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5">
            <Brain className="w-4 h-4 mr-2" />
            AI Analysis Active
          </Badge>
        </div>
      </div>

      {/* Market Sentiment Overview - simple computed summary */}
      <Card className="glass-effect p-6 border-border/50">
        <h2 className="text-xl font-semibold text-foreground mb-6">Market Sentiment Overview</h2>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-3xl font-bold text-primary mb-2">
              {/* positive % */}
              {Math.round(
                (newsItems.filter((n) => n.sentiment === "positive").length / Math.max(1, newsItems.length)) * 100
              )}
              %
            </div>
            <div className="text-sm text-muted-foreground">Positive News</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-muted-foreground mb-2">
              {Math.round(
                (newsItems.filter((n) => n.sentiment === "neutral").length / Math.max(1, newsItems.length)) * 100
              )}
            </div>
            <div className="text-sm text-muted-foreground">Neutral News</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-destructive mb-2">
              {Math.round(
                (newsItems.filter((n) => n.sentiment === "negative").length / Math.max(1, newsItems.length)) * 100
              )}
            </div>
            <div className="text-sm text-muted-foreground">Negative News</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-accent mb-2">
              {newsItems.length ? Math.round(newsItems.reduce((s, it) => s + it.aiConfidence, 0) / newsItems.length) : 0}%
            </div>
            <div className="text-sm text-muted-foreground">AI Confidence (avg)</div>
          </div>
        </div>
      </Card>

      {/* Error */}
      {error ? (
        <Card className="p-4 border-border/50">
          <div className="text-destructive font-medium">Error loading news</div>
          <div className="text-sm text-muted-foreground">{error}</div>
        </Card>
      ) : null}

      {/* News Feed */}
      <div className="space-y-4">
        {newsItems.length === 0 && !error ? (
          <Card className="p-6 border-border/50">
            <div className="text-muted-foreground">No news found.</div>
          </Card>
        ) : (
          newsItems.map((item, index) => (
            <Card key={index} className="glass-effect p-6 border-border/50 hover:border-primary/20 transition-all duration-300">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <Badge variant="outline" className={getSentimentColor(item.sentiment)}>
                    {item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1)}
                  </Badge>
                  <Badge variant="outline" className={`${getImpactColor(item.impact)} border-current/30`}>
                    <AlertTriangle className="w-3 h-3 mr-1" />
                    {item.impact.charAt(0).toUpperCase() + item.impact.slice(1)} Impact
                  </Badge>
                </div>
                <div className="flex items-center gap-2 text-muted-foreground text-sm">
                  <Clock className="w-4 h-4" />
                  {item.time}
                </div>
              </div>

              <h3 className="text-lg font-semibold text-foreground mb-3">{item.title}</h3>

              <p className="text-muted-foreground mb-4 leading-relaxed">{item.summary}</p>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <span className="text-sm text-muted-foreground">Source: {item.source}</span>
                  <div className="flex items-center gap-2">
                    <Brain className="w-4 h-4 text-primary" />
                    <span className="text-sm text-foreground font-medium">AI Confidence: {item.aiConfidence}%</span>
                  </div>
                </div>

                <Button variant="ghost" size="sm" className="text-primary hover:text-primary/80">
                  Read Full Article
                  <ExternalLink className="w-4 h-4 ml-2" />
                </Button>
              </div>
            </Card>
          ))
        )}
      </div>

      {/* Load More (simple refresh here) */}
      <div className="text-center">
        <Button
          variant="outline"
          className="border-primary/30 text-primary hover:bg-primary/10"
          onClick={() => {
            // simple refresh by toggling state: re-run effect by changing key (we use companyQuery / numArticles stable),
            // easiest: reload window or implement a more complex fetch function — keep simple:
            window.location.reload();
          }}
        >
          Load More News
          <TrendingUp className="w-4 h-4 ml-2" />
        </Button>
      </div>
    </div>
  );
}
