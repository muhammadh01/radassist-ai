import { useState, useCallback } from "react";
import "./App.css";

const API_URL = "";
const SAMPLES = [
  { id: 1, src: "/samples/sample1.png", label: "Sample 1" },
  { id: 2, src: "/samples/sample2.png", label: "Sample 2" },
  { id: 3, src: "/samples/sample3.png", label: "Sample 3" },
];

export default function App() {
  const [original, setOriginal] = useState(null);
  const [mask, setMask] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const runPrediction = async (file) => {
    setOriginal(URL.createObjectURL(file));
    setMask(null);
    setStats(null);
    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_URL}/predict`, { method: "POST", headers: { "x-api-key": "3piP0H-jIWXYEkdfXlqKZpByskgnlg-CvDlJf7Ddebs" }, body: formData });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setMask(data.mask);
      setStats(data.stats);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (file) runPrediction(file);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) runPrediction(file);
  }, []);

  const useSample = async (src) => {
    const res = await fetch(src);
    const blob = await res.blob();
    const file = new File([blob], "sample.png", { type: blob.type });
    runPrediction(file);
  };

  return (
    <div className="app">
      <header>
        <div className="logo">
          <span className="logo-icon">⚕</span>
          <h1>RadAssist AI</h1>
        </div>
        <p>Automated lung segmentation for chest X-rays</p>
      </header>

      <div
        className={`dropzone ${dragOver ? "drag-over" : ""} ${loading ? "loading" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <input id="file-input" type="file" accept="image/*" onChange={handleFile} disabled={loading} hidden />
        <label htmlFor="file-input" className="dropzone-content">
          <div className="dropzone-icon">{loading ? "⏳" : "📤"}</div>
          <div className="dropzone-text">
            {loading ? "Analyzing X-ray..." : "Drop X-ray here or click to upload"}
          </div>
          <div className="dropzone-hint">PNG, JPG up to 10MB</div>
        </label>
      </div>

      <div className="samples">
        <span className="samples-label">Or try a sample:</span>
        {SAMPLES.map((s) => (
          <button key={s.id} className="sample-btn" onClick={() => useSample(s.src)} disabled={loading}>
            <img src={s.src} alt={s.label} />
          </button>
        ))}
      </div>

      {error && <div className="error">⚠ {error}</div>}

      {original && (
        <>
          <div className="results">
            <div className="result-card">
              <h3>Original X-ray</h3>
              <img src={original} alt="X-ray" />
            </div>
            <div className="result-card">
              <h3>Segmentation Overlay</h3>
              <div className="overlay-wrapper">
                <img src={original} alt="X-ray" />
                {loading && <div className="scan-line"></div>}
                {mask && <img src={mask} alt="Mask" className="overlay-mask" />}
              </div>
            </div>
          </div>

          {stats && (
            <div className="stats">
              <div className="stat">
                <div className="stat-label">Lung Area</div>
                <div className="stat-value">{stats.lung_area_pct}<span>%</span></div>
              </div>
              <div className="stat">
                <div className="stat-label">Confidence</div>
                <div className="stat-value">{stats.confidence}<span>%</span></div>
              </div>
              <div className="stat">
                <div className="stat-label">Resolution</div>
                <div className="stat-value-text">{stats.resolution}</div>
              </div>
            </div>
          )}
        </>
      )}

      <footer>Educational use only · Not for clinical diagnosis</footer>
    </div>
  );
}
