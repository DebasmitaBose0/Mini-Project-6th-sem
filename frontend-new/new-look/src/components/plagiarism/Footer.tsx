import { ScanLine } from "lucide-react";

export const Footer = () => {
  return (
    <footer className="border-t border-border bg-card">
      <div className="container py-16">
        <div className="grid gap-12 md:grid-cols-5">
          <div className="md:col-span-2">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-sm bg-primary text-primary-foreground">
                <ScanLine className="h-4 w-4" />
              </div>
              <span className="font-display text-xl">Plagiarism<span className="text-accent"> AI</span></span>
            </div>
            <p className="mt-4 max-w-sm text-sm text-muted-foreground">
              An advanced originality engine to support academic and professional integrity.
            </p>
          </div>

          {[
            { title: "Product", links: ["Scanner", "API", "Browser extension", "Word add-in"] },
            { title: "Company", links: ["About", "Research", "Careers", "Press"] },
            { title: "Legal", links: ["Privacy", "Terms", "Security", "DPA"] },
          ].map((c) => (
            <div key={c.title}>
              <div className="mb-4 font-mono text-xs uppercase tracking-widest text-muted-foreground">{c.title}</div>
              <ul className="space-y-3 text-sm">
                {c.links.map((l) => (
                  <li key={l}><a href="#" className="hover:text-accent">{l}</a></li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-16 flex flex-col items-start justify-between gap-4 border-t border-border pt-8 md:flex-row md:items-center">
          <div className="font-mono text-xs text-muted-foreground">© 2026 Plagiarism AI Labs </div>
          <div className="font-display text-sm italic text-muted-foreground">"Empowering Originality."</div>
        </div>
      </div>
    </footer>
  );
};
