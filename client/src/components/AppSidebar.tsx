import { NavLink, useLocation } from "react-router-dom";
import { 
  BarChart3, 
  Brain, 
  Home, 
  LineChart, 
  MessageSquare, 
  Newspaper, 
  Settings,
  TrendingUp,
  Zap
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar";

const navigationItems = [
  { title: "Home", url: "/", icon: Home },
  { title: "Dashboard", url: "/dashboard", icon: BarChart3 },
  { title: "News & Insights", url: "/news", icon: Newspaper },
  { title: "AI Models", url: "/about", icon: Brain },
];

const toolItems = [
  { title: "Sentiment Tracker", url: "/sentiment", icon: TrendingUp },
  { title: "Model Insights", url: "/model-insights", icon: Brain },
];

export function AppSidebar() {
  const { state } = useSidebar();
  const location = useLocation();
  const collapsed = state === "collapsed";

  const isActive = (path: string) => location.pathname === path;

  const getNavClassName = (path: string) => {
    const baseClasses = "flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200";
    if (isActive(path)) {
      return `${baseClasses} bg-primary/10 text-primary border border-primary/20 neon-glow`;
    }
    return `${baseClasses} text-muted-foreground hover:text-foreground hover:bg-accent/50`;
  };

  return (
    <Sidebar className={`glass-effect border-r border-border/50 ${collapsed ? "w-16" : "w-64"}`}>
      <SidebarContent className="p-4">
        {/* Logo Section */}
        <div className="mb-8 px-2">
          {!collapsed ? (
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-primary flex items-center justify-center">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gradient">TrendVision AI</span>
            </div>
          ) : (
            <div className="w-8 h-8 rounded-lg bg-gradient-primary flex items-center justify-center mx-auto">
              <Brain className="w-5 h-5 text-white" />
            </div>
          )}
        </div>

        {/* Main Navigation */}
        <SidebarGroup>
          <SidebarGroupLabel className={collapsed ? "sr-only" : "text-xs uppercase tracking-wider text-muted-foreground mb-2"}>
            Navigation
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navigationItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink to={item.url} className={getNavClassName(item.url)}>
                      <item.icon className="w-5 h-5 flex-shrink-0" />
                      {!collapsed && <span className="font-medium">{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* AI Tools */}
        <SidebarGroup className="mt-6">
          <SidebarGroupLabel className={collapsed ? "sr-only" : "text-xs uppercase tracking-wider text-muted-foreground mb-2"}>
            AI Tools
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {toolItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink to={item.url} className={getNavClassName(item.url)}>
                      <item.icon className="w-5 h-5 flex-shrink-0" />
                      {!collapsed && <span className="font-medium">{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {/* Settings */}
        <div className="mt-auto">
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton asChild>
                <NavLink to="/settings" className={getNavClassName("/settings")}>
                  <Settings className="w-5 h-5 flex-shrink-0" />
                  {!collapsed && <span className="font-medium">Settings</span>}
                </NavLink>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </div>
      </SidebarContent>
    </Sidebar>
  );
}