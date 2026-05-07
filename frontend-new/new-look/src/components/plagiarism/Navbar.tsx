import { ScanLine, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState, useEffect } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

export const Navbar = () => {
  const [isDark, setIsDark] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (document.documentElement.classList.contains("dark")) {
      setIsDark(true);
    }
  }, []);

  const toggleTheme = () => {
    if (document.documentElement.classList.contains("dark")) {
      document.documentElement.classList.remove("dark");
      setIsDark(false);
    } else {
      document.documentElement.classList.add("dark");
      setIsDark(true);
    }
  };

  const handleScanClick = () => {
    if (location.pathname !== "/") {
      navigate("/");
      setTimeout(() => {
        document.getElementById("scan")?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } else {
      document.getElementById("scan")?.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <header className="sticky top-0 z-50 border-b border-border/60 bg-background/80 backdrop-blur-md">
      <div className="container flex h-16 items-center justify-between">
        <Link to="/" className="group flex items-center gap-2 transition-all">
          <div className="flex h-8 w-8 items-center justify-center rounded-sm bg-primary text-primary-foreground transition-all duration-300 group-hover:bg-accent group-hover:shadow-[0_0_15px_hsl(var(--accent))]">
            <ScanLine className="h-4 w-4 transition-transform duration-300 group-hover:scale-110 group-hover:animate-pulse" />
          </div>
          <span className="font-display text-lg font-medium tracking-tight">Plagiarism<span className="text-accent"> AI</span></span>
        </Link>

        <nav className="hidden items-center gap-8 md:flex">
          {["Product", "Database", "Pricing", "Research", "Docs"].map((l) => (
            <Link key={l} to={`/${l.toLowerCase()}`} className="text-sm text-muted-foreground transition-colors hover:text-foreground">
              {l}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="rounded-full w-9 h-9"
            onClick={toggleTheme}
            aria-label="Toggle Theme"
          >
            {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>

          <Button
            size="sm"
            className="rounded-sm"
            onClick={handleScanClick}
          >
            Start scanning
          </Button>
        </div>
      </div>
    </header>
  );
};
