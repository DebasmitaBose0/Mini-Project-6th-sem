import { motion } from "framer-motion";
import { ArrowRight, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

export const Hero = () => {
  return (
    <section className="relative overflow-hidden bg-gradient-hero">
      <div className="bg-grain absolute inset-0 opacity-[0.04] mix-blend-multiply" />
      <div className="container relative grid grid-cols-1 gap-12 py-24 md:py-32 lg:grid-cols-12 lg:gap-8">
        <div className="lg:col-span-7">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-card px-4 py-1.5 text-xs uppercase tracking-[0.2em] text-muted-foreground"
          >
            <Sparkles className="h-3 w-3 text-accent" />
            Advanced AI Originality Engine
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.1 }}
            className="font-display text-[clamp(2.5rem,6vw,5rem)] font-medium leading-[1.1] tracking-tight text-balance"
          >
            Ensure Authenticity,
            <br />
            <span className="text-accent">Empower Originality.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="mt-8 max-w-xl text-lg leading-relaxed text-muted-foreground"
          >
            Plagiarism AI cross-references your writing against billions of web pages and known AI corpora — providing an advanced, precise report to help you maintain academic and professional integrity.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="mt-10 flex flex-wrap items-center gap-4"
          >
            <Button
              size="lg"
              className="group h-12 rounded-sm px-6 text-base"
              onClick={() => document.getElementById("scan")?.scrollIntoView({ behavior: "smooth" })}
            >
              Scan a document
              <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Button>
            <Button
              size="lg"
              variant="ghost"
              className="h-12 rounded-sm px-6 text-base"
              onClick={() => document.getElementById("scan")?.scrollIntoView({ behavior: "smooth" })}
            >
              See a sample report
            </Button>
          </motion.div>

          <div className="mt-12 flex items-baseline gap-8 border-t border-border pt-6">
            {[
              { v: "89B", l: "Pages indexed" },
              { v: "0.4s", l: "Avg. scan / page" },
              { v: "99.2%", l: "Detection accuracy" },
            ].map((s) => (
              <div key={s.l}>
                <div className="font-display text-2xl font-medium tracking-tight">{s.v}</div>
                <div className="mt-1 text-xs uppercase tracking-widest text-muted-foreground">{s.l}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Visual — animated document scan */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.9, delay: 0.2 }}
          className="relative lg:col-span-5"
        >
          <div className="relative overflow-hidden rounded-sm border border-border bg-card shadow-deep">
            <div className="flex items-center justify-between border-b border-border bg-muted/40 px-4 py-2.5">
              <div className="flex items-center gap-1.5">
                <div className="h-2.5 w-2.5 rounded-full bg-risk-high/70" />
                <div className="h-2.5 w-2.5 rounded-full bg-risk-medium/70" />
                <div className="h-2.5 w-2.5 rounded-full bg-risk-low/70" />
              </div>
              <span className="font-mono text-xs text-muted-foreground">thesis_final.docx</span>
              <span className="font-mono text-xs text-accent">SCANNING</span>
            </div>

            <div className="relative px-6 py-8 font-display text-[15px] leading-relaxed">
              <div className="absolute inset-x-0 top-0 h-px animate-scan bg-gradient-to-r from-transparent via-accent to-transparent" />
              <p className="mb-3">
                The fundamental challenge of <span className="highlight-match">distributed consensus lies not in achieving agreement</span>, but in doing so under conditions of partial failure.
              </p>
              <p className="mb-3 text-muted-foreground">
                In practice, <span className="highlight-paraphrase">network partitions force a system to choose</span> between availability and consistency — a tension first formalized by Brewer.
              </p>
              <p className="text-muted-foreground/70">
                Subsequent work has refined this view, suggesting <span className="highlight-match">that latency itself constitutes a third axis</span> of the trade-off space.
              </p>
            </div>

            <div className="border-t border-border bg-muted/30 px-6 py-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-xs uppercase tracking-widest text-muted-foreground">Originality</span>
                <span className="font-display text-2xl font-medium">73<span className="text-muted-foreground text-base">%</span></span>
              </div>
              <div className="h-1 overflow-hidden rounded-full bg-border">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: "73%" }}
                  transition={{ duration: 1.5, delay: 0.8, ease: "easeOut" }}
                  className="h-full bg-gradient-to-r from-accent to-risk-medium"
                />
              </div>
            </div>
          </div>

          <div className="absolute -bottom-6 -left-6 hidden rounded-sm border border-border bg-card px-4 py-3 shadow-editorial md:block">
            <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">Sources matched</div>
            <div className="font-display text-xl">14 documents</div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};
