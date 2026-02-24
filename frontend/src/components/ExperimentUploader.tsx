// src/components/ExperimentUploader.tsx
import React, { useState, useCallback } from 'react';
import './ExperimentUploader.css'; // We'll create this CSS

interface ExperimentUploaderProps {
  onExperimentQueued: () => void; // Callback to notify parent (App/ExperimentList) to refresh
}

const ExperimentUploader: React.FC<ExperimentUploaderProps> = ({ onExperimentQueued }) => {
  const [highlight, setHighlight] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setHighlight(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setHighlight(false);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    // Indicate that the drop is possible
    e.dataTransfer.dropEffect = 'copy';
    setHighlight(true);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setHighlight(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type === 'application/json') {
        uploadFile(file);
      } else {
        setError('Please drop a JSON file.');
        setMessage(null);
      }
      e.dataTransfer.clearData();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (file.type === 'application/json') {
        uploadFile(file);
      } else {
        setError('Please select a JSON file.');
        setMessage(null);
      }
    }
  };

  const uploadFile = useCallback(async (file: File) => {
    setMessage('Uploading experiment config...');
    setError(null);
    const formData = new FormData();
    formData.append('config_file', file);

    try {
      const response = await fetch('http://localhost:8000/queue', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setMessage(`Experiment ${data.id} queued successfully!`);
      onExperimentQueued(); // Notify parent to refresh
      // Clear message after a short delay
      setTimeout(() => setMessage(null), 3000);

    } catch (err) {
      console.error('Error queuing experiment:', err);
      setError(`Failed to queue experiment: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setMessage(null); // Clear success message if there was one
    }
  }, [onExperimentQueued]);


  return (
    <div className="uploader-container">
      <h2>Queue New Experiment</h2>
      <div
        className={`drop-area ${highlight ? 'highlight' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <p>Drag & Drop your JSON config file here</p>
        <p>or</p>
        <input
          type="file"
          id="fileInput"
          accept="application/json"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <label htmlFor="fileInput" className="upload-button">Browse Files</label>
      </div>
      {message && <p className="uploader-message success">{message}</p>}
      {error && <p className="uploader-message error">{error}</p>}
    </div>
  );
};

export default ExperimentUploader;
