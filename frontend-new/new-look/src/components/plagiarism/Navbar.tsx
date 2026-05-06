import { ScanLine } from "lucide-react";
import { Button } from "@/components/ui/button";

export const Navbar = () => {
  return (
    <header className="sticky top-0 z-50 border-b border-border/60 bg-background/80 backdrop-blur-md">
      <div className="container flex h-16 items-center justify-between">
        <a href="#" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-sm bg-primary text-primary-foreground">
            <ScanLine className="h-4 w-4" />
          </div>
          <span className="font-display text-lg font-medium tracking-tight">Plagiarism<span className="text-accent"> AI</span></span>
        </a>

        <nav className="hidden items-center gap-8 md:flex">
          {["Product", "Database", "Pricing", "Research", "Docs"].map((l) => (
            <a key={l} href="#" className="text-sm text-muted-foreground transition-colors hover:text-foreground">
              {l}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Button
            size="sm"
            className="rounded-sm"
            onClick={() => document.getElementById("scan")?.scrollIntoView({ behavior: "smooth" })}
          >
            Start scanning
          </Button>
        </div>
      </div>
    </header>
  );
};
