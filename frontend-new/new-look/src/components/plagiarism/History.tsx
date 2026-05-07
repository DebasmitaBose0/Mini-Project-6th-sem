import { useState, useEffect } from "react";
import { format } from "date-fns";
import { Trash2, Copy, FileText, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

interface HistoryEntry {
  id: string;
  timestamp: string;
  original: string;
  rewritten: string;
  similarity: number;
}

export const History = () => {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetchHistory = async () => {
    try {
      const response = await fetch("http://localhost:8000/history");
      const data = await response.json();
      setHistory(data.reverse()); // Show newest first
    } catch (error) {
      console.error(error);
    }
  };

  useEffect(() => {
    fetchHistory();
    // Setting up a basic interval to refresh history every few seconds for smooth UX
    const interval = setInterval(fetchHistory, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await fetch(`http://localhost:8000/history/${id}`, { method: "DELETE" });
      toast.success("Entry deleted from history.");
      fetchHistory();
    } catch (error) {
      toast.error("Failed to delete entry.");
    }
  };

  const handleCopy = (text: string, e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard!");
  };

  if (history.length === 0) return null;

  return (
    <section id="history" className="container py-24 bg-muted/10 border-t border-border">
      <div className="mx-auto max-w-5xl">
        <div className="mb-12">
          <h2 className="font-display text-3xl font-medium tracking-tight">Recent Scans</h2>
          <p className="text-muted-foreground mt-2">View, copy, and manage your plagiarism checking history.</p>
        </div>

        <div className="space-y-4">
          {history.map((entry) => {
            const isExpanded = expandedId === entry.id;
            return (
              <div key={entry.id} className="rounded-sm border border-border bg-card overflow-hidden">
                <div 
                  className="flex items-center justify-between p-4 cursor-pointer hover:bg-muted/30 transition-colors"
                  onClick={() => setExpandedId(isExpanded ? null : entry.id)}
                >
                  <div className="flex items-center gap-4">
                    <div className="bg-muted p-2 rounded">
                      <FileText className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <div>
                      <div className="font-medium truncate max-w-xs md:max-w-md">
                        {entry.original.slice(0, 50)}...
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {format(new Date(entry.timestamp), "PPpp")}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="hidden md:block px-3 py-1 bg-risk-low/10 text-accent text-xs font-mono rounded">
                      {(entry.similarity * 100).toFixed(1)}% Sim
                    </div>
                    {isExpanded ? <ChevronUp className="h-5 w-5 text-muted-foreground" /> : <ChevronDown className="h-5 w-5 text-muted-foreground" />}
                  </div>
                </div>

                {isExpanded && (
                  <div className="p-6 border-t border-border bg-muted/10 grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <div className="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-2 flex justify-between items-center">
                        <span>Original Text</span>
                      </div>
                      <div className="bg-background border border-border rounded p-4 text-sm whitespace-pre-wrap max-h-60 overflow-y-auto">
                        {entry.original}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-bold uppercase tracking-widest text-accent mb-2 flex items-center justify-between">
                        <span>Rewritten Output</span>
                        <div className="flex gap-2">
                          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={(e) => handleCopy(entry.rewritten, e)} title="Copy rewritten text">
                            <Copy className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-6 w-6 text-destructive hover:text-destructive" onClick={(e) => handleDelete(entry.id, e)} title="Delete entry">
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                      <div className="bg-background border border-accent/20 rounded p-4 text-sm whitespace-pre-wrap max-h-60 overflow-y-auto font-serif">
                        {entry.rewritten}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};