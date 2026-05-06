import { Navbar } from "@/components/plagiarism/Navbar";
import { Hero } from "@/components/plagiarism/Hero";
import { Uploader } from "@/components/plagiarism/Uploader";
import { Footer } from "@/components/plagiarism/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main>
        <Hero />
        <Uploader />
      </main>
      <Footer />
    </div>
  );
};

export default Index;
