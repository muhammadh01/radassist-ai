import { useState } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

export default function App() {
  const [original, setOriginal] = useState(null);
  const [mask, setMask] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setOriginal(URL.createObjectURL(file));
    setMask(null);
    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Prediction failed");

      const blob = await res.blob();
      setMask(URL.createObjectURL(blob));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>RadAssist AI</h1>
        <p>Automated lung segmentation for chest X-rays</p>
      </header>

      <div className="upload-box">
        <label htmlFor="file-input" className="upload-btn">
          {loading ? "Analyzing..." : "Upload X-ray"}
        </label>
        <input
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleUpload}
          disabled={loading}
          hidden
        />
      </div>

      {error && <div className="error">{error}</div>}

      {original && (
        <div className="results">
          <div className="result-card">
            <h3>Original</h3>
            <img src={original} alt="X-ray" />
          </div>
          <div className="result-card">
            <h3>Lung Segmentation</h3>
            {loading ? (
              <div className="loading">Processing...</div>
            ) : mask ? (
              <img src={mask} alt="Mask" />
            ) : null}
          </div>
        </div>
      )}

      <footer>
        Educational use only · Not for clinical diagnosis
      </footer>
    </div>
  );
}
