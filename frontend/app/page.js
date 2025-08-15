'use client';

import { useState } from 'react';
import Link from 'next/link'; 

export default function HomePage() {
  const [landSize, setLandSize] = useState('');
  const [crop, setCrop] = useState('');
  const [plan, setPlan] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setPlan('');

    try {
      const response = await fetch(process.env.NEXT_PUBLIC_YIELDWISE_URL, {
        
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ land_size: parseFloat(landSize), crop: crop }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        // Split the plan into bullet points for better display
        setPlan(data.plan.split('* ').filter(item => item.trim() !== ''));
      }
    } catch (err) {
      setError('Failed to fetch the financial plan. Please ensure the backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="container">
      <header>
        <h1>🌾 YieldWise</h1>
        <p>Your AI-Powered Agricultural Financial Advisor</p>
        <nav className="navigation">
            <Link href="/groundtruth">Switch to GroundTruth AI</Link> | <Link href="/vision">FieldScout AI</Link>
        </nav>
      </header>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="landSize">Land Size (in acres)</label>
          <input
            id="landSize"
            type="number"
            value={landSize}
            onChange={(e) => setLandSize(e.target.value)}
            placeholder="e.g., 5"
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="crop">Crop to be Planted</label>
          <input
            id="crop"
            type="text"
            value={crop}
            onChange={(e) => setCrop(e.target.value)}
            placeholder="e.g., Tomato"
            required
          />
        </div>
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Generating...' : 'Generate Plan'}
        </button>
      </form>

      {error && <div className="result error">{error}</div>}

      {plan && (
        <div className="result">
          <h2>Your Financial Plan</h2>
          <ul>
            {plan.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        </div>
      )}
    </main>
  );
}