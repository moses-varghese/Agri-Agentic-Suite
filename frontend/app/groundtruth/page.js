'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import VoiceRecorder from '../components/VoiceRecorder';

export default function GroundTruthPage() {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: 'Hello! How can I help you with your agricultural questions today? Ask me a question with your voice or by typing' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [audioObject, setAudioObject] = useState(null); // 👈 Store the audio object
  const [isPlaying, setIsPlaying] = useState(false); // 👈 Track playback state
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  // Stop any playing audio when the component unmounts
  useEffect(() => {
    return () => {
      if (audioObject) {
        audioObject.pause();
      }
    };
  }, [audioObject]);


  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(process.env.NEXT_PUBLIC_GROUNDTRUTH_AI_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      });

      if (!response.ok) throw new Error('Network response was not ok');
      
      const data = await response.json();
      const aiMessage = { sender: 'ai', text: data.answer };
      setMessages(prev => [...prev, aiMessage]);

    } catch (err) {
      const errorMessage = { sender: 'ai', text: 'Sorry, I am having trouble connecting to the AI service. Please try again.' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };





  const handleVoiceSubmit = async (audioFile) => {
    setIsLoading(true);
    if (audioObject) audioObject.pause();

    const formData = new FormData();
    formData.append('audio_file', audioFile);

    try {
      setMessages(prev => [...prev, { sender: 'user', text: '🎤 (You sent a voice note)' }]);
      
      const response = await fetch(process.env.NEXT_PUBLIC_GROUNDTRUTH_AI_VOICE_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // If the server returns an error, try to parse it as JSON
        const errorData = await response.json();
        throw new Error(errorData.error || 'Network response was not ok');
      }

      // --- THIS IS THE FIX ---
      // Check the response type before processing
      const contentType = response.headers.get("content-type");

      if (contentType && contentType.includes("audio/mpeg")) {
        // Handle the successful audio response
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const newAudio = new Audio(audioUrl);
        
        newAudio.onplay = () => setIsPlaying(true);
        newAudio.onended = () => setIsPlaying(false);
        newAudio.play();
        
        setAudioObject(newAudio);
        setMessages(prev => [...prev, { sender: 'ai', text: '🔊 (AI is responding...)' }]);
      } else {
        // Handle an unexpected (but successful) JSON response
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Received an unexpected response from the server.');
      }

    } catch (err) {
      // Now, this will display the actual error message from the backend
      const errorMessage = { sender: 'ai', text: `Sorry, an error occurred: ${err.message}` };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // --- NEW: Playback Control Functions ---
  const handleStop = () => {
    if (audioObject) {
      audioObject.pause();
      audioObject.currentTime = 0; // Rewind to the beginning
      setIsPlaying(false);
    }
  };

  const handleReplay = () => {
    if (audioObject) {
      audioObject.currentTime = 0; // Rewind to the beginning
      audioObject.play();
    }
  };

  return (
    <main className="container">
       <header>
        <h1>🗣️ GroundTruth AI</h1>
        <p>Your General-Purpose Agricultural AI Assistant</p>
        <nav className="navigation">
            <Link href="/">Switch to YieldWise</Link> | <Link href="/vision">FieldScout AI</Link>
        </nav>
      </header>

      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`chat-message ${msg.sender}`}>
            <p>{msg.text}</p>
          </div>
        ))}
        {isLoading && (
          <div className="chat-message ai">
            <p className="loading-dots"><span>.</span><span>.</span><span>.</span></p>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* --- NEW: Audio Control Buttons --- */}
      {audioObject && (
        <div className="audio-controls">
          <button onClick={handleReplay} disabled={isPlaying}>Replay Last</button>
          <button onClick={handleStop} disabled={!isPlaying}>Stop</button>
        </div>
      )}

      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>Send</button>
        {/*VoiceRecorder component here */}
        <VoiceRecorder onStop={handleVoiceSubmit} />
      </form>
    </main>
  );
}