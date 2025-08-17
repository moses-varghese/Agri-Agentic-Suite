'use client';
import { useState, useEffect } from 'react';
import MicRecorder from 'mic-recorder-to-mp3';

// Initialize the recorder
const recorder = new MicRecorder({ bitRate: 128 });

export default function VoiceRecorder({ onStop }) {
  const [isRecording, setIsRecording] = useState(false);
  const [isBlocked, setIsBlocked] = useState(false);

  useEffect(() => {
    // Check for microphone permissions
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(() => setIsBlocked(false))
      .catch(() => setIsBlocked(true));
  }, []);

  const startRecording = () => {
    if (isBlocked) {
      alert('Microphone access is blocked. Please allow access in your browser settings.');
      return;
    }
    recorder.start().then(() => {
      setIsRecording(true);
    });
  };

  const stopRecording = () => {
    recorder.stop().getMp3().then(([buffer, blob]) => {
      const file = new File(buffer, 'voice-message.mp3', {
        type: blob.type,
        lastModified: Date.now()
      });
      onStop(file); // Pass the recorded audio file to the parent component
      setIsRecording(false);
    });
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="voice-recorder">
      <button
        type="button"
        className={`record-button ${isRecording ? 'recording' : ''}`}
        onClick={toggleRecording}
        disabled={isBlocked}
      >
        {isRecording ? 'Stop' : '🎤 Record'}
      </button>
      {isBlocked && <p className="error">Microphone access denied.</p>}
    </div>
  );
}