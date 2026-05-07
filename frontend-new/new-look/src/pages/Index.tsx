import { Navbar } from "@/components/plagiarism/Navbar";
import { Hero } from "@/components/plagiarism/Hero";
import { Uploader } from "@/components/plagiarism/Uploader";
import { History } from "@/components/plagiarism/History";
import { Footer } from "@/components/plagiarism/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main>
        <Hero />
        <div className="container py-8 flex flex-col-reverse lg:flex-row gap-8 items-start">
          <div className="w-full lg:w-1/3 shrink-0">
            <History />
          </div>
          <div className="w-full lg:w-2/3">
            <Uploader />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default Index;
