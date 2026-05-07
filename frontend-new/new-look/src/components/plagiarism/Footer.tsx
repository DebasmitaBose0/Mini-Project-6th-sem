import { ScanLine } from "lucide-react";
import { Link } from "react-router-dom";

export const Footer = () => {
  return (
    <footer className="border-t border-border bg-card">
      <div className="container py-16">
        <div className="grid gap-12 md:grid-cols-5">
          <div className="md:col-span-2">
            <div className="group flex w-max cursor-pointer items-center gap-2 transition-all">
              <div className="flex h-8 w-8 items-center justify-center rounded-sm bg-primary text-primary-foreground transition-all duration-300 group-hover:bg-accent group-hover:shadow-[0_0_15px_hsl(var(--accent))]">
                <ScanLine className="h-4 w-4 transition-transform duration-300 group-hover:scale-110 group-hover:animate-pulse" />
              </div>
              <span className="font-display text-xl">Plagiarism<span className="text-accent"> AI</span></span>
            </div>
            <p className="mt-4 max-w-sm text-sm text-muted-foreground">
              An advanced originality engine to support academic and professional integrity.
            </p>
          </div>

          <div>
            <div className="mb-4 font-mono text-xs uppercase tracking-widest text-muted-foreground">Product</div>
            <ul className="space-y-3 text-sm">
              <li><button onClick={() => window.scrollTo(0, 0)} className="hover:text-accent">Scanner</button></li>
              <li><Link to="/api-info" className="hover:text-accent">API</Link></li>
            </ul>
          </div>
          <div>
            <div className="mb-4 font-mono text-xs uppercase tracking-widest text-muted-foreground">Company</div>
            <ul className="space-y-3 text-sm">
              <li><Link to="/about" className="hover:text-accent">About</Link></li>
              <li><Link to="/research" className="hover:text-accent">Research</Link></li>
              <li><Link to="/careers" className="hover:text-accent">Careers</Link></li>
              <li><Link to="/press" className="hover:text-accent">Press</Link></li>
            </ul>
          </div>
          <div>
            <div className="mb-4 font-mono text-xs uppercase tracking-widest text-muted-foreground">Legal</div>
            <ul className="space-y-3 text-sm">
              <li><Link to="/privacy" className="hover:text-accent">Privacy</Link></li>
              <li><Link to="/terms" className="hover:text-accent">Terms</Link></li>
              <li><Link to="/security" className="hover:text-accent">Security</Link></li>
              <li><Link to="/dpa" className="hover:text-accent">DPA</Link></li>
            </ul>
          </div>
        </div>

        <div className="mt-16 flex flex-col items-start justify-between gap-4 border-t border-border pt-8 md:flex-row md:items-center">
          <div className="font-mono text-xs text-muted-foreground">© 2026 Plagiarism AI - Made by Debasmita, Manisha, Joita, Suchitra | All Rights Reserved</div>
          <div className="font-display text-sm italic text-muted-foreground">"Empowering Originality."</div>
        </div>
      </div>
    </footer>
  );
};
