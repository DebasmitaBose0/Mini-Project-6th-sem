import { ScanLine } from "lucide-react";
import { Link } from "react-router-dom";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

const FooterLink = ({ to, children, description, isButton = false }: { to?: string, children: React.ReactNode, description: string, isButton?: boolean }) => {
  const content = isButton ? (
    <button onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} className="hover:text-accent transition-colors">
      {children}
    </button>
  ) : (
    <Link to={to || "#"} className="hover:text-accent transition-colors">
      {children}
    </Link>
  );

  return (
    <TooltipProvider>
      <Tooltip delayDuration={300}>
        <TooltipTrigger asChild>
          <li>{content}</li>
        </TooltipTrigger>
        <TooltipContent side="right" className="max-w-[200px] text-xs">
          <p>{description}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export const Footer = () => {
  return (
    <footer className="border-t border-border bg-card mt-auto">
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
              An advanced originality engine designed by Debasmita and her teammates to support academic and professional integrity across global research.
            </p>

          </div>

          <div>
            <div className="mb-4 font-mono text-xs uppercase tracking-widest text-muted-foreground">Product</div>
            <ul className="space-y-3 text-sm">
              <FooterLink isButton description="Launch our advanced AI scanner to analyze your text and identify potential plagiarism.">
                Scanner
              </FooterLink>
              <FooterLink to="/api-info" description="Integrate our plagiarism detection engine directly into your own applications via our robust API.">
                API
              </FooterLink>
            </ul>
          </div>

          <div>
            <div className="mb-4 font-mono text-xs uppercase tracking-widest text-muted-foreground">Company</div>
            <ul className="space-y-3 text-sm">
              <FooterLink to="/about" description="Discover the mission, vision, and the expert team behind Plagiarism AI's technology.">
                About
              </FooterLink>
              <FooterLink to="/research" description="Explore our peer-reviewed research on linguistic patterns and AI-generated content detection.">
                Research
              </FooterLink>
              <FooterLink to="/careers" description="Join our growing team of AI researchers and developers to build the future of originality.">
                Careers
              </FooterLink>
              <FooterLink to="/press" description="Access our official media kit, brand assets, and latest company announcements.">
                Press
              </FooterLink>
            </ul>
          </div>

          <div>
            <div className="mb-4 font-mono text-xs uppercase tracking-widest text-muted-foreground">Legal</div>
            <ul className="space-y-3 text-sm">
              <FooterLink to="/privacy" description="Review our strict data protection protocols and how we handle your sensitive content.">
                Privacy
              </FooterLink>
              <FooterLink to="/terms" description="Read our terms of service and acceptable use policies for the Plagiarism AI platform.">
                Terms
              </FooterLink>
              <FooterLink to="/security" description="Learn about our enterprise-grade security measures and data encryption standards.">
                Security
              </FooterLink>
              <FooterLink to="/dpa" description="Our Data Processing Agreement for institutional and organizational compliance.">
                DPA
              </FooterLink>
            </ul>
          </div>
        </div>

        <div className="mt-16 flex flex-col items-start justify-between gap-4 border-t border-border pt-8 md:flex-row md:items-center">
          <div className="font-mono text-xs text-muted-foreground">© 2026 Plagiarism AI - Created by Debasmita, Manisha, Joita, Suchitra | All Rights Reserved</div>
          <div className="font-display text-sm italic text-muted-foreground">"Empowering Originality."</div>
        </div>
      </div>
    </footer>
  );
};
