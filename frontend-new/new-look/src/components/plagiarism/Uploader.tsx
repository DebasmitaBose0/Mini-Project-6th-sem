import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { FileText, Link2, Loader2, Type, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";

const tabs = [
  { id: "text", label: "Paste text", icon: Type },
];

export const Uploader = () => {
  const [active, setActive] = useState("text");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [fileName, setFileName] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<{ similarity: number; plagiarism_percent: number; rewritten: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleScan = async () => {
    if (active === "text" && !text.trim()) return toast.error("Please paste some text to scan.");
    if (active === "file" && !fileName) return toast.error("Please upload a file first.");
    if (active === "url" && !url.trim()) return toast.error("Please enter a URL to scan.");
    
    setScanning(true);
    setResult(null);
    try {
      const response = await fetch("http://localhost:8000/rewrite", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });
      
      if (!response.ok) {
        throw new Error("Failed to scan document");
      }
      
      const data = await response.json();
      setResult(data);
      toast.success(`Scan complete — ${data.plagiarism_percent.toFixed(2)}% similarity to original.`);
    } catch (error) {
      console.error(error);
      toast.error("Failed to connect to backend server.");
    } finally {
      setScanning(false);
    }
  };

  const handleFile = (f: File | null) => {
    if (!f) return;
    setFileName(f.name);
    toast.success(`${f.name} ready to scan.`);
  };
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;

  return (
    <section id="scan" className="container py-24">
      <div className="mx-auto max-w-5xl">
        <div className="mb-12 text-center">
          <div className="mb-3 font-mono text-xs uppercase tracking-[0.3em] text-accent">— Document Analysis</div>
          <h2 className="font-display text-4xl font-medium tracking-tight md:text-5xl">
            Submit Your Text
          </h2>
          <p className="mt-4 text-muted-foreground">Select a method to begin scanning and rewriting your content.</p>
        </div>

        <div className="overflow-hidden rounded-sm border border-border bg-card shadow-editorial">
          <div className="flex border-b border-border">
            {tabs.map((t) => {
              const Icon = t.icon;
              const isActive = active === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setActive(t.id)}
                  className={`relative flex flex-1 items-center justify-center gap-2 px-6 py-4 text-sm font-medium transition-colors ${
                    isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {t.label}
                  {isActive && (
                    <motion.div layoutId="tab-underline" className="absolute inset-x-0 bottom-0 h-0.5 bg-accent" />
                  )}
                </button>
              );
            })}
          </div>

          <div className="p-8">
            {active === "text" && (
              <div>
                <Textarea
                  placeholder="Paste your text here. Plagiarism AI yields the best results with substantial paragraphs."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  className="min-h-[280px] resize-none rounded-sm border-border bg-background font-display text-base leading-relaxed focus-visible:ring-accent"
                />
                <div className="mt-3 flex justify-start font-mono text-xs text-muted-foreground">
                  <span>{words.toLocaleString()} words</span>
                </div>
              </div>
            )}

            {active === "file" && (
              <div className="flex flex-col items-center justify-center rounded-sm border-2 border-dashed border-border bg-background py-20 text-center">
                <FileText className="mb-4 h-10 w-10 text-muted-foreground" />
                <div className="font-display text-xl">{fileName ?? "Drop your document here"}</div>
                <p className="mt-2 text-sm text-muted-foreground">
                  PDF, DOCX, TXT, RTF — up to 50MB
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.doc,.docx,.txt,.rtf"
                  className="hidden"
                  onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
                />
                <Button variant="outline" className="mt-6 rounded-sm" onClick={() => fileInputRef.current?.click()}>
                  {fileName ? "Choose another" : "Browse files"}
                </Button>
              </div>
            )}

            {active === "url" && (
              <div className="space-y-3">
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com/article"
                  className="w-full rounded-sm border border-border bg-background px-4 py-3 font-mono text-sm focus:border-accent focus:outline-none"
                />
                <p className="text-xs text-muted-foreground">We'll fetch the page and analyze its readable content.</p>
              </div>
            )}
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
            <Button size="lg" className="rounded-sm" onClick={handleScan} disabled={scanning}>
              {scanning && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {scanning ? "Scanning…" : "Run scan"}
            </Button>
          </div>
          
          {result && (
            <div className="p-8 border-t border-border bg-card">
              <h3 className="mb-4 font-display text-2xl">Analysis & Rewrite Results</h3>
              <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="rounded-sm border border-border p-4 bg-risk-low/10">
                  <div className="text-sm text-muted-foreground flex items-center justify-between">
                    Similarity to Original
                    <span className="text-xs uppercase bg-background px-2 py-0.5 rounded border border-border">Generated vs Input</span>
                  </div>
                  <div className="font-display text-4xl mt-2 text-accent">
                    {result.plagiarism_percent.toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    A lower percentage means the new text is significantly restructured and rephrased compared to your input, ensuring high originality.
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
