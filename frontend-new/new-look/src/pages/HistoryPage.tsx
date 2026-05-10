import { Navbar } from "@/components/plagiarism/Navbar";
import { History } from "@/components/plagiarism/History";
import { Footer } from "@/components/plagiarism/Footer";

export default function HistoryPage() {
  return (
    <div className="min-h-screen flex flex-col bg-background font-sans antialiased text-foreground">
      <Navbar />
      
      <main className="flex-1 container py-12 flex flex-col items-center">
        <div className="w-full max-w-4xl space-y-6">
          <div className="text-center mb-8">
            <h1 className="font-display text-4xl font-medium tracking-tight">
              Scan History
            </h1>
            <p className="mt-4 text-muted-foreground">
              Review your recent document analyses and AI rewritten texts.
            </p>
          </div>
          <History />
        </div>
      </main>

      <Footer />
    </div>
  );
}
