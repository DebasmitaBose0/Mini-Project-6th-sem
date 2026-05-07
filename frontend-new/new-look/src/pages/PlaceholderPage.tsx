import { useParams, useLocation, Link } from "react-router-dom";
import { Navbar } from "@/components/plagiarism/Navbar";
import { Footer } from "@/components/plagiarism/Footer";
import { Button } from "@/components/ui/button";
import { MoveLeft } from "lucide-react";

export default function PlaceholderPage() {
  const location = useLocation();
  const pathName = location.pathname.replace("/", "");
  const title = pathName.charAt(0).toUpperCase() + pathName.slice(1);

  return (
    <div className="min-h-screen flex flex-col bg-background font-sans antialiased text-foreground">
      <Navbar />
      
      <main className="flex-1 flex flex-col items-center justify-center p-6 text-center">
        <div className="max-w-2xl space-y-6">
          <div className="inline-block rounded-full bg-accent/10 px-4 py-1.5 font-mono text-xs text-accent">
            Under Construction
          </div>
          <h1 className="font-display text-4xl sm:text-5xl font-medium tracking-tight">
            {title}
          </h1>
          <p className="text-lg text-muted-foreground">
            We're currently working on this page. Check back soon for updates on our {title.toLowerCase()} policies and information.
          </p>
          <div className="pt-8">
            <Button asChild variant="outline">
              <Link to="/">
                <MoveLeft className="mr-2 h-4 w-4" /> Back to Home
              </Link>
            </Button>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}