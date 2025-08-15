import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  TrendingUp, 
  Clock,
  ExternalLink,
  Brain,
  AlertTriangle
} from "lucide-react";

export default function NewsPage() {
  const newsItems = [
    {
      title: "Federal Reserve Signals Potential Rate Cut in Q2 2024",
      summary: "AI analysis suggests 78% probability of bullish market reaction. Tech stocks likely to benefit most significantly.",
      source: "Reuters",
      time: "2 hours ago",
      sentiment: "positive",
      impact: "high",
      aiConfidence: 78
    },
    {
      title: "Bitcoin ETF Approval Sparks Institutional Interest",
      summary: "Cryptocurrency markets showing strong momentum. AI predicts 15-20% upside potential in next 30 days.",
      source: "Bloomberg",
      time: "4 hours ago",
      sentiment: "positive",
      impact: "medium",
      aiConfidence: 85
    },
    {
      title: "Tech Earnings Season: Mixed Results Expected",
      summary: "AI sentiment analysis indicates cautious optimism. FAANG stocks may see increased volatility.",
      source: "Financial Times",
      time: "6 hours ago",
      sentiment: "neutral",
      impact: "medium",
      aiConfidence: 72
    },
    {
      title: "Oil Prices Surge on Geopolitical Tensions",
      summary: "Energy sector outlook turns bullish. AI models predict sustained price increase for commodities.",
      source: "CNBC",
      time: "8 hours ago",
      sentiment: "negative",
      impact: "high",
      aiConfidence: 82
    },
    {
      title: "Chinese Manufacturing Data Shows Recovery",
      summary: "Asian markets likely to benefit. AI forecasts positive spillover effects to global equity markets.",
      source: "Wall Street Journal",
      time: "12 hours ago",
      sentiment: "positive",
      impact: "medium",
      aiConfidence: 67
    }
  ];

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'text-primary border-primary/30 bg-primary/5';
      case 'negative': return 'text-destructive border-destructive/30 bg-destructive/5';
      default: return 'text-accent border-accent/30 bg-accent/5';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'text-destructive';
      case 'medium': return 'text-accent';
      default: return 'text-muted-foreground';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient mb-2">News & Insights</h1>
          <p className="text-muted-foreground">AI-powered market news analysis and sentiment tracking</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5">
            <Brain className="w-4 h-4 mr-2" />
            AI Analysis Active
          </Badge>
        </div>
      </div>

      {/* Market Sentiment Overview */}
      <Card className="glass-effect p-6 border-border/50">
        <h2 className="text-xl font-semibold text-foreground mb-6">Market Sentiment Overview</h2>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-3xl font-bold text-primary mb-2">+68%</div>
            <div className="text-sm text-muted-foreground">Positive News</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-muted-foreground mb-2">18%</div>
            <div className="text-sm text-muted-foreground">Neutral News</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-destructive mb-2">14%</div>
            <div className="text-sm text-muted-foreground">Negative News</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-accent mb-2">87%</div>
            <div className="text-sm text-muted-foreground">AI Accuracy</div>
          </div>
        </div>
      </Card>

      {/* News Feed */}
      <div className="space-y-4">
        {newsItems.map((item, index) => (
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
        ))}
      </div>

      {/* Load More */}
      <div className="text-center">
        <Button variant="outline" className="border-primary/30 text-primary hover:bg-primary/10">
          Load More News
          <TrendingUp className="w-4 h-4 ml-2" />
        </Button>
      </div>
    </div>
  );
}