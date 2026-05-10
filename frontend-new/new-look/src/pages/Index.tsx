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
        <div className="container py-8 flex flex-col gap-12 items-center">
          <div className="w-full max-w-4xl">
            <Uploader />
          </div>
        </div>

      </main>
      <Footer />
    </div>
  );
};

export default Index;
