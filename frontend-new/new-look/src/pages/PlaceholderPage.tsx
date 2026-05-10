import { useParams, useLocation, Link } from "react-router-dom";
import { Navbar } from "@/components/plagiarism/Navbar";
import { Footer } from "@/components/plagiarism/Footer";
import { Button } from "@/components/ui/button";
import { MoveLeft } from "lucide-react";

export default function PlaceholderPage() {
  const location = useLocation();
  const pathName = location.pathname.replace("/", "");
  const title = pathName.charAt(0).toUpperCase() + pathName.slice(1);

  const pageDetails: Record<string, React.ReactNode> = {
    "About": (
      <div className="space-y-4 text-left mt-6">
        <p><strong>Plagiarism AI</strong> was founded by Debasmita and her dedicated team of developers to tackle one of the most pressing issues in modern academia and content creation: originality.</p>
        <p>Our mission is to empower writers, researchers, and educators with an advanced, open-source tool that not only detects similarities against billions of web pages but also offers intelligent, context-aware rewriting capabilities.</p>
        <p>Built as a final year mini-project, this system represents our commitment to accessible, high-performance software engineering utilizing modern AI frameworks like Ollama, FastAPI, and React.</p>
      </div>
    ),
    "Research": (
      <div className="space-y-4 text-left mt-6">
        <p>Our core technology relies on advanced Natural Language Processing (NLP) techniques, including TF-IDF vectorization and Cosine Similarity mapping, to detect structural and semantic overlap between documents.</p>
        <p>Furthermore, our integration with local Large Language Models (LLMs) like Llama 3 allows us to experiment with local-first, privacy-preserving AI paraphrasing. This ensures that sensitive academic documents never leave your machine.</p>
        <p>Current research efforts are focused on improving the speed of local inference and expanding our document parsing capabilities to include more complex file structures and multi-modal inputs.</p>
      </div>
    ),
    "Careers": (
      <div className="space-y-4 text-left mt-6">
        <p>Join our growing team of AI researchers, software engineers, and designers who are passionate about building the future of originality and academic integrity.</p>
        <p>While we are currently a student-led initiative, we are always looking to collaborate with open-source contributors. Whether you specialize in Python backend optimization, React frontend design, or LLM prompting, there is a place for you here.</p>
        <p>Check out our GitHub repository to see our current open issues and contribution guidelines.</p>
      </div>
    ),
    "Press": (
      <div className="space-y-4 text-left mt-6">
        <p>For press inquiries, interviews, or access to our media kit, please reach out to our communications team.</p>
        <p>Plagiarism AI has been developed as an open-source alternative to expensive, proprietary plagiarism checkers, aiming to democratize access to essential academic tools.</p>
        <p>Brand assets, including our logo, typography guidelines, and high-resolution screenshots, will be made available here shortly.</p>
      </div>
    ),
    "Privacy": "Review our strict data protection protocols and how we handle your sensitive content. Because our engine runs locally via Ollama, your text never leaves your computer.",
    "Terms": "Read our terms of service and acceptable use policies. By using this open-source platform, you agree to utilize it responsibly and ethically.",
    "Security": "Learn about our security measures. All document processing happens on your local machine, guaranteeing enterprise-grade security and zero data leaks.",
    "Dpa": "Our Data Processing Agreement for institutional compliance.",
    "Pricing": "This platform is completely free to use and it is 100% open source. There are no hidden fees or subscription tiers.",
    "Docs": "Read our comprehensive documentation and guides to understand how the system works, how to set up the local LLM, and how to use the API."
  };

  const content = pageDetails[title] || `We're currently working on this page. Check back soon for updates on our ${title.toLowerCase()} policies and information.`;

  return (
    <div className="min-h-screen flex flex-col bg-background font-sans antialiased text-foreground">
      <Navbar />
      
      <main className="flex-1 flex flex-col items-center justify-center p-6 text-center">
        <div className="max-w-3xl space-y-6">
          <div className="inline-block rounded-full bg-accent/10 px-4 py-1.5 font-mono text-xs text-accent">
            {title} Section
          </div>
          <h1 className="font-display text-4xl sm:text-5xl font-medium tracking-tight">
            {title}
          </h1>
          <div className="text-lg text-muted-foreground leading-relaxed">
            {content}
          </div>
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