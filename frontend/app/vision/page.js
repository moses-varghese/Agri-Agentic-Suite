'use client';

import { useState } from 'react';
import Link from 'next/link';

export default function VisionPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [diagnosis, setDiagnosis] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setDiagnosis(null);
      setError('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an image first.');
      return;
    }
    setIsLoading(true);
    setError('');
    setDiagnosis(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch(process.env.NEXT_PUBLIC_FIELDSCOUT_AI_URL, {
        method: 'POST',
        body: formData, // No 'Content-Type' header needed, browser sets it for FormData
      });

      if (!response.ok) throw new Error('Network response was not ok');
      
      const data = await response.json();
      setDiagnosis(data);

    } catch (err) {
      setError('Failed to get diagnosis. Please ensure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="container">
      <header>
        <h1>🌿 FieldScout AI</h1>
        <p>Your AI-Powered Crop Disease Diagnostic Tool</p>
        <nav className="navigation">
            <Link href="/">YieldWise</Link> | <Link href="/groundtruth">Groundtruth AI</Link>
        </nav>
      </header>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="imageUpload">Upload an image of a plant leaf</label>
          <input
            id="imageUpload"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />
        </div>

        {preview && (
          <div className="image-preview">
            <img src={preview} alt="Image preview" />
          </div>
        )}

        <button type="submit" disabled={isLoading || !file}>
          {isLoading ? 'Diagnosing...' : 'Diagnose'}
        </button>
      </form>

      {error && <div className="result error">{error}</div>}
      
      {diagnosis && (
        <div className="result">
          <h2>Diagnosis Result</h2>
          <p><strong>Condition:</strong> {diagnosis.diagnosis}</p>
          <p><strong>Confidence:</strong> {diagnosis.confidence}</p>
          <p><strong>Recommendation:</strong> {diagnosis.recommendation}</p>
        </div>
      )}
    </main>
  );
}