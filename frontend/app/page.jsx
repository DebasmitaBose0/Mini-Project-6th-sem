import { useState } from "react";

export default function Home() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);

  const handleRewrite = async () => {
    const res = await fetch("http://localhost:8000/rewrite", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: input })
    });

    const data = await res.json();
    setResult(data);
  };

  return (
    <div className="container">
      <h1>PlagiarismAI</h1>

      <textarea
        placeholder="Enter text..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />

      <button onClick={handleRewrite}>Rewrite</button>

      {result && (
        <div className="output">
          <div>
            <h3>Original</h3>
            <p>{result.original}</p>
          </div>

          <div>
            <h3>Rewritten</h3>
            <p>{result.rewritten}</p>
          </div>

          <div className="metrics">
            <p>Similarity: {(result.similarity * 100).toFixed(2)}%</p>
            <p>Plagiarism: {result.plagiarism_percent.toFixed(2)}%</p>
          </div>
        </div>
      )}
    </div>
  );
}