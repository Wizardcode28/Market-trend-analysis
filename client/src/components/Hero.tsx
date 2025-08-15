import { ArrowRight, Brain, TrendingUp, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useNavigate } from "react-router-dom";
import heroImage from "@/assets/hero-bg.jpg";

export function Hero() {
  const navigate = useNavigate();

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background Image with Overlay */}
      <div 
        className="absolute inset-0 z-0"
        style={{
          backgroundImage: `url(${heroImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat'
        }}
      >
        <div className="absolute inset-0 bg-dark-bg/80 backdrop-blur-sm"></div>
      </div>

      {/* Floating elements */}
      <div className="absolute top-20 right-20 w-64 h-64 bg-gradient-primary rounded-full opacity-10 blur-3xl animate-float"></div>
      <div className="absolute bottom-20 left-20 w-48 h-48 bg-accent rounded-full opacity-10 blur-3xl animate-float" style={{ animationDelay: '1s' }}></div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 text-center">
        {/* Badge */}
        <Badge variant="outline" className="mb-6 text-primary border-primary/30 bg-primary/5 text-sm">
          <Zap className="w-4 h-4 mr-2" />
          Powered by Advanced AI
        </Badge>

        {/* Main Heading */}
        <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
          <span className="text-gradient">TrendVision AI</span>
          <br />
          <span className="text-foreground">Predicts Markets</span>
        </h1>

        {/* Subtitle */}
        <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto">
          Harness the power of artificial intelligence to analyze and predict market trends 
          in stocks, crypto, and commodities with unprecedented accuracy.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
          <Button 
            size="lg" 
            className="bg-gradient-primary hover:opacity-90 text-white px-8 py-6 text-lg neon-glow"
            onClick={() => navigate('/dashboard')}
          >
            Start Analyzing
            <ArrowRight className="w-5 h-5 ml-2" />
          </Button>
          <Button 
            variant="outline" 
            size="lg" 
            className="border-primary/30 text-primary hover:bg-primary/10 px-8 py-6 text-lg"
            onClick={() => navigate('/about')}
          >
            Learn About Our AI
            <Brain className="w-5 h-5 ml-2" />
          </Button>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <Card className="glass-effect p-6 text-center border-border/50 hover:border-primary/30 transition-all duration-300">
            <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center mx-auto mb-4">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold mb-2 text-foreground">Real-Time Predictions</h3>
            <p className="text-muted-foreground text-sm">
              Get AI-powered predictions with confidence levels for short and long-term market movements.
            </p>
          </Card>

          <Card className="glass-effect p-6 text-center border-border/50 hover:border-primary/30 transition-all duration-300">
            <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center mx-auto mb-4">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold mb-2 text-foreground">Advanced Analytics</h3>
            <p className="text-muted-foreground text-sm">
              Comprehensive sentiment analysis, news impact, and technical indicators in one platform.
            </p>
          </Card>

          <Card className="glass-effect p-6 text-center border-border/50 hover:border-primary/30 transition-all duration-300">
            <div className="w-12 h-12 rounded-lg bg-gradient-primary flex items-center justify-center mx-auto mb-4">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold mb-2 text-foreground">Instant Insights</h3>
            <p className="text-muted-foreground text-sm">
              Real-time data feeds and lightning-fast analysis for quick decision making.
            </p>
          </Card>
        </div>
      </div>
    </section>
  );
}