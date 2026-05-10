import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { Loader2, Type } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";

const tabs = [
  { id: "text", label: "Paste text", icon: Type },
];

export const Uploader = () => {
  const [active, setActive] = useState("text");
  const [text, setText] = useState("");
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<{ similarity: number; plagiarism_percent: number; rewritten: string } | null>(null);

  const handleScan = async () => {
    const formData = new FormData();
    
    if (!text.trim()) return toast.error("Please paste some text to scan.");
    const wordCount = text.trim().split(/\s+/).length;
    if (wordCount > 400) return toast.error("Please limit your text to 400 words.");
    formData.append("text", text);
    
    setScanning(true);
    setResult(null);
    try {
      const response = await fetch("http://localhost:8000/rewrite", {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Failed to scan document");
      }
      
      const data = await response.json();
      setResult(data);
      toast.success(`Scan complete — ${data.plagiarism_percent.toFixed(2)}% similarity to original.`);
    } catch (error: any) {
      console.error(error);
      toast.error(error.message || "Failed to connect to backend server.");
    } finally {
      setScanning(false);
    }
  };

  const words = text.trim() ? text.trim().split(/\s+/).length : 0;

  return (
    <section id="scan" className="container pb-24 pt-8">
      <div className="mx-auto max-w-4xl">
        <div className="mb-12 text-center">
          <div className="mb-3 font-mono text-xs uppercase tracking-[0.3em] text-accent">— Document Analysis</div>
          <h2 className="font-display text-4xl font-medium tracking-tight md:text-5xl">
            Submit Your Text
          </h2>
          <p className="mt-4 text-muted-foreground">Paste your content below to begin scanning and rewriting.</p>
        </div>

        <div className="overflow-hidden rounded-sm border border-border bg-card shadow-editorial">
          <div className="flex border-b border-border">
            {tabs.map((t) => {
              const Icon = t.icon;
              const isActive = active === t.id;
              return (
                <div
                  key={t.id}
                  className="relative flex flex-1 items-center justify-center gap-2 px-6 py-4 text-sm font-medium text-foreground bg-muted/5"
                >
                  <Icon className="h-4 w-4" />
                  {t.label}
                  <motion.div layoutId="tab-underline" className="absolute inset-x-0 bottom-0 h-0.5 bg-accent" />
                </div>
              );
            })}
          </div>

          <div className="p-8">
            <div>
              <Textarea
                placeholder="Paste your text here. Plagiarism AI yields the best results with substantial paragraphs."
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="min-h-[280px] resize-none rounded-sm border-border bg-background font-display text-base leading-relaxed focus-visible:ring-accent"
              />
              <div className={`mt-3 flex justify-start font-mono text-xs ${words > 400 ? 'text-destructive font-bold text-red-500' : 'text-muted-foreground'}`}>
                <span>{words.toLocaleString()} / 400 words</span>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between border-t border-border bg-muted/20 px-8 py-5">
            <div className="flex items-center gap-6 text-xs text-muted-foreground">
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="accent-accent" /> Detect AI-generated
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="accent-accent" /> Find paraphrasing
              </label>
              <label className="hidden items-center gap-2 sm:flex">
                <input type="checkbox" className="accent-accent" /> Exclude citations
              </label>
            </div>
            <Button size="lg" className="rounded-sm" onClick={handleScan} disabled={scanning || words > 400}>
              {scanning && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {scanning ? "Scanning…" : "Run scan"}
            </Button>
          </div>
          
          {scanning && (
            <div className="p-8 border-t border-border bg-card flex flex-col items-center justify-center animate-pulse space-y-4">
              <Loader2 className="h-12 w-12 animate-spin text-accent" />
              <p className="text-muted-foreground font-medium animate-pulse">Analyzing and rewriting your content. Please wait...</p>
            </div>
          )}

          {!scanning && result && (
            <div className="p-8 border-t border-border bg-card">
              <h3 className="mb-4 font-display text-2xl">Analysis & Rewrite Results</h3>
              <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="rounded-sm border border-border p-4 bg-risk-low/10">
                  <div className="text-sm text-muted-foreground flex items-center justify-between">
                    Similarity to Original
                    <span className="text-xs uppercase bg-background px-2 py-0.5 rounded border border-border">Overlap</span>
                  </div>
                  <div className="font-display text-4xl mt-2 text-destructive">
                    {result.plagiarism_percent.toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Amount of structural and verbatim overlap found between the new and original text.
                  </p>
                </div>

                <div className="rounded-sm border border-border p-4 bg-accent/5">
                  <div className="text-sm text-muted-foreground flex items-center justify-between">
                    Plagiarism Removal
                    <span className="text-xs uppercase bg-background px-2 py-0.5 rounded border border-border">Success Rate</span>
                  </div>
                  <div className="font-display text-4xl mt-2 text-accent">
                    {(100 - result.plagiarism_percent).toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    Percentage of content that has been successfully transformed into original material.
                  </p>
                </div>

              </div>
              <div className="space-y-6">
                <div className="rounded-sm border border-border p-6 bg-background relative">
                  <div className="absolute top-0 right-0 rounded-bl-sm rounded-tr-sm bg-muted px-3 py-1 text-xs font-medium text-muted-foreground shadow-sm">
                    Original Text
                  </div>
                  <div className="text-sm font-medium mb-3 text-muted-foreground uppercase tracking-widest">Your Input</div>
                  <p className="whitespace-pre-wrap text-[15px] leading-relaxed font-sans text-muted-foreground">{text}</p>
                </div>

                <div className="rounded-sm border-2 border-accent/20 p-6 bg-muted/10 relative">
                  <div className="absolute top-0 right-0 rounded-bl-sm rounded-tr-sm bg-accent px-3 py-1 text-xs font-medium text-white shadow-sm">
                    AI Rewritten Output
                  </div>
                  <div className="text-sm font-medium mb-3 text-accent uppercase tracking-widest">Plagiarism-Free Content</div>
                  <div className="text-[16px] leading-relaxed font-serif text-foreground">
                    <p className="font-bold mb-4">Here's a completely rewritten text that removes any trace of plagiarism while preserving the original meaning, tone, and key information:</p>
                    <p className="whitespace-pre-wrap">{result.rewritten.replace(/\*\*.*?\*\*/g, '').replace(/Here's a completely rewritten text.*?:\n*/gi, '').replace(/Note: I've used a different structure.*?original\.\n*/gi, '').trim()}</p>
                    <p className="font-bold mt-4">Note: I've used a different structure, vocabulary, and sentence flow to produce an original text that maintains the same meaning and key information as the original.</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
