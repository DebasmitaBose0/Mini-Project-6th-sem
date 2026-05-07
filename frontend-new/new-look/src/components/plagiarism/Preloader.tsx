import { useEffect, useState } from "react";

export const Preloader = () => {
  const [show, setShow] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShow(false);
    }, 2000);
    return () => clearTimeout(timer);
  }, []);

  if (!show) return null;

  return (
    <div className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-background/80 backdrop-blur-sm transition-opacity duration-500">
      <div className="relative flex h-24 w-24 items-center justify-center">
        <div className="absolute h-full w-full rounded-full border-4 border-accent/20 border-t-accent animate-spin" style={{ animationDuration: '1.2s' }}></div>
        <div className="absolute h-16 w-16 rounded-full border-4 border-primary/20 border-b-primary animate-spin" style={{ animationDuration: '0.9s', animationDirection: 'reverse' }}></div>
      </div>
      <div className="mt-8 font-mono text-sm font-medium tracking-widest text-foreground animate-pulse">
        Starting Plagiarism AI...
      </div>
    </div>
  );
};