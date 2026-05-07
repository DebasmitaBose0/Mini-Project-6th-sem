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
    <section id="history" className="bg-muted/10 border border-border rounded-lg shadow-sm h-fit">
      <div className="p-6">
        <div className="mb-6">
          <h2 className="font-display text-2xl font-medium tracking-tight">Recent Scans</h2>
          <p className="text-sm text-muted-foreground mt-1 text-balance">View, copy, and manage your plagiarism checking history.</p>
        </div>

        <div className="space-y-3">
          {history.map((entry) => {
            const isExpanded = expandedId === entry.id;
            return (
              <div key={entry.id} className="rounded-md border border-border bg-card overflow-hidden">
                <div 
                  className="flex flex-col gap-3 p-4 cursor-pointer hover:bg-muted/30 transition-colors"
                  onClick={() => setExpandedId(isExpanded ? null : entry.id)}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-3 w-full overflow-hidden">
                      <div className="bg-muted p-2 rounded shrink-0">
                        <FileText className="h-4 w-4 text-muted-foreground" />
                      </div>
                      <div className="overflow-hidden">
                        <div className="font-medium text-sm truncate w-full">
                          {entry.original.slice(0, 40)}...
                        </div>
                        <div className="text-xs text-muted-foreground mt-0.5">
                          {format(new Date(entry.timestamp), "MMM d, h:mm a")}
                        </div>
                      </div>
                    </div>
                    <div className="shrink-0 mt-1">
                      {isExpanded ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between text-xs px-1">
                    <div className="px-2 py-0.5 bg-risk-low/10 text-accent font-mono rounded inline-block">
                      {(entry.similarity * 100).toFixed(1)}% Sim
                    </div>
                  </div>
                </div>

                {isExpanded && (
                  <div className="p-4 border-t border-border bg-muted/10 flex flex-col gap-4">
                    <div>
                      <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1.5 flex justify-between items-center">
                        <span>Original Text</span>
                      </div>
                      <div className="bg-background border border-border rounded p-3 text-xs whitespace-pre-wrap max-h-40 overflow-y-auto">
                        {entry.original}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] font-bold uppercase tracking-widest text-accent mb-1.5 flex items-center justify-between">
                        <span>Rewritten Output</span>
                        <div className="flex gap-1.5">
                          <Button variant="ghost" size="icon" className="h-5 w-5" onClick={(e) => handleCopy(entry.rewritten, e)} title="Copy rewritten text">
                            <Copy className="h-3 w-3" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-5 w-5 text-destructive hover:text-destructive" onClick={(e) => handleDelete(entry.id, e)} title="Delete entry">
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                      <div className="bg-background border border-accent/20 rounded p-3 text-xs whitespace-pre-wrap max-h-40 overflow-y-auto font-serif">
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