'use client';

import { useState } from 'react';
import Link from 'next/link';
import ReactMarkdown from 'react-markdown';

export default function VisionPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analysis, setAnalysis] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [analysisMode, setAnalysisMode] = useState('disease_diagnosis'); // New state for the mode

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setAnalysis('');
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
    setAnalysis('');

    const formData = new FormData();
    formData.append('image', file);
    formData.append('analysis_mode', analysisMode); // Send the selected mode

    try {
      const response = await fetch(process.env.NEXT_PUBLIC_FIELDSCOUT_AI_URL, {
        method: 'POST',
        body: formData, // No 'Content-Type' header needed, browser sets it for FormData
      });

      if (!response.ok) throw new Error('Network response was not ok');
      
      const data = await response.json();
      setAnalysis(data.analysis);

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
            <Link href="/">YieldWise</Link> | <Link href="/groundtruth">GroundTruth AI</Link>
        </nav>
      </header>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="analysisMode">Select Analysis Type</label>
          <select 
            id="analysisMode" 
            value={analysisMode} 
            onChange={(e) => setAnalysisMode(e.target.value)}
          >
            <option value="disease_diagnosis">Disease Diagnosis</option>
            <option value="crop_identification">Crop Identification</option>
            <option value="field_health_analysis">Field Health Analysis</option>
          </select>
        </div>


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
      
      {analysis && (
        <div className="result">
          <h2>AI Analysis</h2>
          <ReactMarkdown>{analysis}</ReactMarkdown>
        </div>
      )}
    </main>
  );
}