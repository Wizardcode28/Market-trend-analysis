import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  Brain, 
  Cpu, 
  Database, 
  TrendingUp,
  Shield,
  Zap,
  Users,
  Award
} from "lucide-react";

export default function AboutPage() {
  const aiModels = [
    {
      name: "TrendPredict Neural Network",
      description: "Deep learning model trained on 15+ years of market data",
      accuracy: 87,
      specialty: "Price Prediction",
      dataPoints: "2.5M+",
      icon: Brain
    },
    {
      name: "SentimentAnalyzer GPT",
      description: "Natural language processing for news and social sentiment",
      accuracy: 92,
      specialty: "Sentiment Analysis",
      dataPoints: "1.8M+",
      icon: Cpu
    },
    {
      name: "VolatilityEngine ML",
      description: "Machine learning model for volatility and risk assessment",
      accuracy: 79,
      specialty: "Risk Analysis",
      dataPoints: "3.2M+",
      icon: Database
    }
  ];

  const features = [
    {
      title: "Real-Time Processing",
      description: "Process millions of data points in milliseconds",
      icon: Zap
    },
    {
      title: "Secure & Private",
      description: "Enterprise-grade security with encrypted data",
      icon: Shield
    },
    {
      title: "Continuously Learning",
      description: "Models update and improve with new market data",
      icon: TrendingUp
    },
    {
      title: "Expert Validated",
      description: "Verified by financial experts and data scientists",
      icon: Award
    }
  ];

  return (
    <div className="p-6 space-y-8">
      {/* Header */}
      <div className="text-center max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gradient mb-4">Our AI Models</h1>
        <p className="text-xl text-muted-foreground">
          Transparent, powerful, and continuously evolving artificial intelligence 
          that powers your market analysis and predictions.
        </p>
      </div>

      {/* Model Overview */}
      <div className="grid lg:grid-cols-3 gap-6">
        {aiModels.map((model, index) => (
          <Card key={index} className="glass-effect p-6 border-border/50 hover:border-primary/20 transition-all duration-300">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center">
                <model.icon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-foreground">{model.name}</h3>
                <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5 text-xs">
                  {model.specialty}
                </Badge>
              </div>
            </div>

            <p className="text-muted-foreground mb-6">{model.description}</p>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-muted-foreground">Accuracy Rate</span>
                  <span className="text-sm font-medium text-foreground">{model.accuracy}%</span>
                </div>
                <Progress value={model.accuracy} className="h-2" />
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Training Data Points</span>
                <span className="text-sm font-medium text-primary">{model.dataPoints}</span>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* How It Works */}
      <Card className="glass-effect p-8 border-border/50">
        <h2 className="text-2xl font-bold text-foreground mb-6 text-center">How Our AI Works</h2>
        
        <div className="grid md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="w-16 h-16 rounded-full bg-gradient-primary flex items-center justify-center mx-auto mb-4">
              <Database className="w-8 h-8 text-white" />
            </div>
            <h3 className="font-semibold text-foreground mb-2">Data Ingestion</h3>
            <p className="text-sm text-muted-foreground">
              Real-time market data from multiple sources including exchanges, news, and social media
            </p>
          </div>

          <div className="text-center">
            <div className="w-16 h-16 rounded-full bg-gradient-primary flex items-center justify-center mx-auto mb-4">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h3 className="font-semibold text-foreground mb-2">AI Processing</h3>
            <p className="text-sm text-muted-foreground">
              Advanced neural networks analyze patterns and correlations in the data
            </p>
          </div>

          <div className="text-center">
            <div className="w-16 h-16 rounded-full bg-gradient-primary flex items-center justify-center mx-auto mb-4">
              <TrendingUp className="w-8 h-8 text-white" />
            </div>
            <h3 className="font-semibold text-foreground mb-2">Prediction</h3>
            <p className="text-sm text-muted-foreground">
              Generate predictions with confidence levels and supporting analysis
            </p>
          </div>

          <div className="text-center">
            <div className="w-16 h-16 rounded-full bg-gradient-primary flex items-center justify-center mx-auto mb-4">
              <Users className="w-8 h-8 text-white" />
            </div>
            <h3 className="font-semibold text-foreground mb-2">Delivery</h3>
            <p className="text-sm text-muted-foreground">
              Present insights through intuitive dashboards and real-time alerts
            </p>
          </div>
        </div>
      </Card>

      {/* Key Features */}
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-6 text-center">Why Trust Our AI?</h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <Card key={index} className="glass-effect p-6 border-border/50 text-center hover:border-primary/20 transition-all duration-300">
              <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center mx-auto mb-4">
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-semibold text-foreground mb-2">{feature.title}</h3>
              <p className="text-sm text-muted-foreground">{feature.description}</p>
            </Card>
          ))}
        </div>
      </div>

      {/* Performance Stats */}
      <Card className="glass-effect p-8 border-border/50">
        <h2 className="text-2xl font-bold text-foreground mb-6 text-center">Performance Metrics</h2>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
          <div>
            <div className="text-4xl font-bold text-primary mb-2">87%</div>
            <div className="text-sm text-muted-foreground">Overall Accuracy</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-accent mb-2">15M+</div>
            <div className="text-sm text-muted-foreground">Predictions Made</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-primary mb-2">24/7</div>
            <div className="text-sm text-muted-foreground">Uptime</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-accent mb-2">50ms</div>
            <div className="text-sm text-muted-foreground">Response Time</div>
          </div>
        </div>
      </Card>
    </div>
  );
}