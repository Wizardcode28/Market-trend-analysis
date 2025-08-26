import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Brain, Target, BarChart3, TrendingUp, AlertTriangle } from "lucide-react";

export default function ModelInsightsPage() {
  const modelMetrics = {
    training: {
      rmse: "1.0004",
      mae: "0.6035", 
      r2: "0.9972",
      sampleSize: "160"
    },
    testing: {
      rmse: "1.5323",
      mae: "1.2218",
      r2: "0.9733",
      sampleSize: "40"
    }
  };

  const featureImportance = [
    { feature: "20-Day Moving Average", importance: 95 },
    { feature: "Trading Volume", importance: 88 },
    { feature: "RSI (14-day)", importance: 82 },
    { feature: "50-Day Moving Average", importance: 78 },
    { feature: "Market Sentiment", importance: 72 },
    { feature: "Economic Indicators", importance: 68 },
    { feature: "Sector Performance", importance: 64 },
    { feature: "Options Flow", importance: 58 },
    { feature: "News Sentiment", importance: 52 },
    { feature: "Analyst Ratings", importance: 45 }
  ];

  const modelInsights = [
    { title: "Market Volatility Alert", description: "Increased volatility detected in crypto markets due to regulatory news.", severity: "warning" },
    { title: "Tech Stocks Momentum", description: "AI predicts continued upward momentum in major tech stocks for next 48 hours.", severity: "positive" },
    { title: "Gold Support Level", description: "Gold approaching key support level at $2,030. Watch for potential reversal.", severity: "info" },
    { title: "Energy Sector Outlook", description: "Model indicates potential energy sector rotation based on geopolitical factors.", severity: "warning" },
    { title: "Cryptocurrency Patterns", description: "Pattern recognition shows Bitcoin following historical pre-halving behavior.", severity: "positive" },
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gradient mb-2">Model Insights</h1>
          <p className="text-muted-foreground">AI model performance analytics and predictions</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse-glow mr-2"></div>
            Model Active
          </Badge>
          <Button variant="outline" size="sm">
            <Brain className="w-4 h-4 mr-2" />
            Update Model
          </Button>
        </div>
      </div>

      {/* Model Insights */}
      <Card className="glass-effect p-6 border-border/50">
        <h2 className="text-xl font-semibold text-foreground mb-6">AI Insights & Predictions</h2>
        
        <div className="space-y-4">
          {modelInsights.map((insight, index) => (
            <div key={index} className="p-4 rounded-lg bg-card/30 border border-border/20">
              <div className="flex items-start gap-3">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  insight.severity === 'warning' ? 'bg-destructive/20 text-destructive' :
                  insight.severity === 'positive' ? 'bg-primary/20 text-primary' :
                  'bg-accent/20 text-accent'
                }`}>
                  {insight.severity === 'warning' ? <AlertTriangle className="w-4 h-4" /> : 
                   insight.severity === 'positive' ? <TrendingUp className="w-4 h-4" /> : <Brain className="w-4 h-4" />}
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

      {/* Model Performance Details */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card className="glass-effect p-6 border-border/50">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-10 h-10 rounded-lg bg-gradient-primary flex items-center justify-center">
              <Target className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-xl font-semibold text-foreground">Model Performance Details</h2>
          </div>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-destructive mb-4 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-destructive"></div>
                Training Metrics:
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">RMSE:</span>
                  <span className="font-medium text-foreground">{modelMetrics.training.rmse}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">MAE:</span>
                  <span className="font-medium text-foreground">{modelMetrics.training.mae}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">R² Score:</span>
                  <span className="font-medium text-foreground">{modelMetrics.training.r2}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Sample Size:</span>
                  <span className="font-medium text-foreground">{modelMetrics.training.sampleSize}</span>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="font-medium text-primary mb-4 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-primary"></div>
                Testing Metrics:
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">RMSE:</span>
                  <span className="font-medium text-foreground">{modelMetrics.testing.rmse}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">MAE:</span>
                  <span className="font-medium text-foreground">{modelMetrics.testing.mae}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">R² Score:</span>
                  <span className="font-medium text-foreground">{modelMetrics.testing.r2}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Sample Size:</span>
                  <span className="font-medium text-foreground">{modelMetrics.testing.sampleSize}</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-6 p-4 rounded-lg bg-primary/10 border border-primary/20">
            <h3 className="font-medium text-primary mb-2 flex items-center gap-2">
              <Target className="w-4 h-4" />
              Model Interpretation
            </h3>
            <p className="text-sm text-primary/80">✓ Excellent model performance! High accuracy predictions.</p>
          </div>
        </Card>

        <Card className="glass-effect p-6 border-border/50">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-10 h-10 rounded-lg bg-gradient-primary flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-xl font-semibold text-foreground">Feature Importance</h2>
          </div>
          
          <div className="mb-4">
            <h3 className="font-medium text-foreground mb-4">Top 10 Most Important Features</h3>
          </div>
          
          <div className="space-y-3">
            {featureImportance.map((feature, index) => (
              <div key={index} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground truncate">{feature.feature}</span>
                  <span className="text-foreground font-medium">{feature.importance}%</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-gradient-primary h-2 rounded-full transition-all duration-500" 
                    style={{ width: `${feature.importance}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Model Configuration */}
      <Card className="glass-effect p-6 border-border/50">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-foreground">Model Configuration</h2>
          <Button variant="outline" size="sm">
            Edit Configuration
          </Button>
        </div>
        
        <div className="grid md:grid-cols-3 gap-6">
          <div>
            <h3 className="font-medium text-foreground mb-3">Model Type</h3>
            <Badge variant="outline" className="mb-2">Random Forest Regressor</Badge>
            <p className="text-sm text-muted-foreground">Ensemble learning method for robust predictions</p>
          </div>
          
          <div>
            <h3 className="font-medium text-foreground mb-3">Update Frequency</h3>
            <Badge variant="outline" className="mb-2">Real-time</Badge>
            <p className="text-sm text-muted-foreground">Model updates every 5 minutes with new market data</p>
          </div>
          
          <div>
            <h3 className="font-medium text-foreground mb-3">Data Sources</h3>
            <div className="space-y-1">
              <Badge variant="outline" className="text-xs">Market Data</Badge>
              <Badge variant="outline" className="text-xs">News Sentiment</Badge>
              <Badge variant="outline" className="text-xs">Social Media</Badge>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}