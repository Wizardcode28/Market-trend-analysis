import { Menu, Bell, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Badge } from "@/components/ui/badge";

export function Header() {
  return (
    <header className="h-16 glass-effect border-b border-border/50 flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <SidebarTrigger className="text-muted-foreground hover:text-foreground" />
        <div className="hidden md:flex items-center gap-2">
          <Badge variant="outline" className="text-primary border-primary/30 bg-primary/5">
            Live Market Data
          </Badge>
          <div className="w-2 h-2 rounded-full bg-primary animate-pulse-glow"></div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <Button variant="ghost" size="sm" className="relative">
          <Bell className="w-5 h-5" />
          <span className="absolute -top-1 -right-1 w-3 h-3 bg-primary rounded-full text-[10px] text-primary-foreground flex items-center justify-center">
            3
          </span>
        </Button>
        
        <Button variant="ghost" size="sm">
          <User className="w-5 h-5" />
        </Button>
      </div>
    </header>
  );
}