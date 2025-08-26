import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Globe, AlertTriangle, Activity } from "lucide-react";

export default function SentimentTrackerPage() {
  const sentimentData = [
    { 
      title: "EU Carbon Tax Impact", 
      description: "European Union's new carbon border tax affecting Tesla's European operations, causing -2.3% dip in TSLA stock.", 
      severity: "warning",
      region: "EU",
      impact: "TSLA -2.3%"
    },
    { 
      title: "China Trade Relations", 
      description: "Improved US-China trade discussions boosting Apple's supply chain confidence, contributing to +1.8% AAPL surge.", 
      severity: "positive",
      region: "China",
      impact: "AAPL +1.8%"
    },
    { 
      title: "Fed Interest Rate Decision", 
      description: "Federal Reserve's dovish stance on interest rates driving broad market optimism and crypto rally.", 
      severity: "positive",
      region: "US",
      impact: "BTC +5.2%"
    },
    { 
      title: "UK Energy Crisis", 
      description: "Britain's energy policy uncertainty impacting renewable energy stocks and gold as safe haven.", 
      severity: "warning",
      region: "UK",
      impact: "GOLD +0.8%"
    },
    {
      title: "Japan Monetary Policy",
      description: "Bank of Japan's continued ultra-loose monetary policy supporting yen carry trades and tech stock investments.",
      severity: "positive",
      region: "Japan",
      impact: "NIKKEI +2.1%"
    },
    {
      title: "Brazil Political Uncertainty",
      description: "Brazilian political tensions affecting commodity prices and emerging market confidence.",
      severity: "warning", 
      region: "Brazil",
      impact: "EWZ -1.5%"
    }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient mb-2">Global Sentiment Tracker</h1>
          <p className="text-muted-foreground">Foreign policy & economic decisions impact analysis</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse-glow mr-2"></div>
            Live Analysis
          </Badge>
        </div>
      </div>

      {/* Global Sentiment Tracker */}
      <Card className="glass-effect p-6 border-border/50">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center">
            <Globe className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-foreground">Global Sentiment Tracker</h2>
            <p className="text-muted-foreground">Foreign policy & economic decisions impact analysis</p>
          </div>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          {sentimentData.map((sentiment, index) => (
            <div key={index} className="p-4 rounded-lg bg-card/30 border border-border/20">
              <div className="flex items-start gap-3">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  sentiment.severity === 'warning' ? 'bg-destructive/20 text-destructive' :
                  sentiment.severity === 'positive' ? 'bg-primary/20 text-primary' :
                  'bg-accent/20 text-accent'
                }`}>
                  {sentiment.severity === 'warning' ? <AlertTriangle className="w-4 h-4" /> : <Activity className="w-4 h-4" />}
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="font-medium text-foreground">{sentiment.title}</h3>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">{sentiment.region}</Badge>
                      <Badge variant={sentiment.severity === 'positive' ? 'default' : 'destructive'} className="text-xs">
                        {sentiment.impact}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground">{sentiment.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Regional Impact Summary */}
      <div className="grid md:grid-cols-3 gap-6">
        <Card className="glass-effect p-6 border-border/50">
          <h3 className="text-lg font-semibold text-foreground mb-4">Most Impacted Regions</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Europe</span>
              <Badge variant="destructive" className="text-xs">High Impact</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Asia Pacific</span>
              <Badge variant="default" className="text-xs">Moderate</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Americas</span>
              <Badge variant="default" className="text-xs">Moderate</Badge>
            </div>
          </div>
        </Card>

        <Card className="glass-effect p-6 border-border/50">
          <h3 className="text-lg font-semibold text-foreground mb-4">Sector Analysis</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Technology</span>
              <span className="text-primary font-medium">+2.1%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Energy</span>
              <span className="text-destructive font-medium">-1.8%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Financials</span>
              <span className="text-primary font-medium">+0.9%</span>
            </div>
          </div>
        </Card>

        <Card className="glass-effect p-6 border-border/50">
          <h3 className="text-lg font-semibold text-foreground mb-4">Risk Assessment</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Geopolitical Risk</span>
              <Badge variant="destructive" className="text-xs">High</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Economic Risk</span>
              <Badge variant="outline" className="text-xs">Medium</Badge>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Market Volatility</span>
              <Badge variant="outline" className="text-xs">Medium</Badge>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}