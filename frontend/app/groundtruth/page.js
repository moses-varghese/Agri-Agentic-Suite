'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';

export default function QueryPage() {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: 'Hello! How can I help you with your agricultural questions today?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

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

  return (
    <main className="container">
       <header>
        <h1>🗣️ Groundtruth AI</h1>
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

      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>Send</button>
      </form>
    </main>
  );
}