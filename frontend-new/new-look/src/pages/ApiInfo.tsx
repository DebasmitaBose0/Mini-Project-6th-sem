import { Navbar } from "@/components/plagiarism/Navbar";
import { Footer } from "@/components/plagiarism/Footer";

const ApiInfo = () => {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navbar />
      <main className="flex-1 container py-24 mx-auto max-w-4xl">
        <h1 className="font-display text-4xl font-medium tracking-tight md:text-5xl mb-6">API Information</h1>
        
        <div className="space-y-6 text-muted-foreground leading-relaxed">
          <p className="text-lg">
            Our Plagiarism AI API leverages <strong>Ollama</strong> under the hood to ensure robust, localized, and highly accurate document similarity checking, as well as intelligent paraphrasing to maintain content originality.
          </p>

          <div className="bg-card border border-border rounded-lg p-8 space-y-4">
            <h2 className="font-display text-2xl text-foreground">Future Improvements & Architecture</h2>
            <p>
              Right now, our internal systems execute AI queries over high-performance local Ollama-hosted models (such as Llama 3 or Mistral). In the future, this endpoint will be strictly securely gated.
            </p>
            <div className="p-4 bg-muted/30 border-l-4 border-accent italic text-sm">
              Note: In future versions, <strong>API keys</strong> and strict rate limiting will be required for third-party consumers wanting to leverage this tool computationally via REST endpoints.
            </div>
            <p>
              Developers will be able to integrate our engine directly into their CI/CD pipelines, publishing workflows, or LMS grading tools to ensure plagiarism-free confidence out of the box.
            </p>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default ApiInfo;
