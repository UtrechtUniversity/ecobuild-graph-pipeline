// src/components/AddExperimentRun.tsx
import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import './ExperimentUpload.css';

const AddExperimentRun: React.FC = () => {
  const [highlight, setHighlight] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  const addFiles = (newFiles: FileList | null) => {
    if (!newFiles) return;
    setFiles((prev) => [...prev, ...Array.from(newFiles)]);
    setMessage(null);
    setError(null);
  };

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
    e.dataTransfer.dropEffect = 'copy';
    setHighlight(true);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setHighlight(false);
    addFiles(e.dataTransfer.files);
    e.dataTransfer.clearData();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    addFiles(e.target.files);
  };

  const removeFile = (name: string) => {
    setFiles((prev) => prev.filter((f) => f.name !== name));
  };

  const handleSubmit = useCallback(async () => {
    if (files.length === 0) {
      setError('Add at least one file from a knowledge-extraction run (e.g. *_labels.json, *_extraction.json, *_raw.md, *_report.txt).');
      return;
    }

    setUploading(true);
    setMessage('Uploading experiment run...');
    setError(null);

    const formData = new FormData();
    files.forEach((file) => formData.append('files', file, file.name));

    try {
      const response = await fetch('http://localhost:8000/experiments/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setMessage(`Experiment run registered as ${data.id}.`);
      setFiles([]);
      navigate(`/experiments/${data.id}`);
    } catch (err) {
      console.error('Error uploading experiment run:', err);
      setError(`Failed to upload experiment run: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setMessage(null);
    } finally {
      setUploading(false);
    }
  }, [files, navigate]);

  return (
    <div className="uploader-container">
      <h2>Add Experiment Run</h2>
      <p>
        Drag &amp; drop the output files from a knowledge-extraction run
        (e.g. <code>*_labels.json</code>, <code>*_extraction.json</code>,
        <code>*_raw.md</code>, <code>*_report.txt</code>) to register them as
        a completed experiment.
      </p>
      <div
        className={`drop-area ${highlight ? 'highlight' : ''}`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <p>Drag &amp; Drop extraction output files here</p>
        <p>or</p>
        <input
          type="file"
          id="experimentFilesInput"
          multiple
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <label htmlFor="experimentFilesInput" className="upload-button">Browse Files</label>
      </div>

      {files.length > 0 && (
        <ul style={{ textAlign: 'left', marginTop: '15px' }}>
          {files.map((file) => (
            <li key={file.name}>
              {file.name}{' '}
              <button onClick={() => removeFile(file.name)} className="remove-button">Remove</button>
            </li>
          ))}
        </ul>
      )}

      <button
        onClick={handleSubmit}
        disabled={uploading || files.length === 0}
        className="upload-button"
        style={{ marginTop: '15px', border: 'none' }}
      >
        {uploading ? 'Uploading...' : 'Register Experiment Run'}
      </button>

      {message && <p className="uploader-message success">{message}</p>}
      {error && <p className="uploader-message error">{error}</p>}
    </div>
  );
};

export default AddExperimentRun;
