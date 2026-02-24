// src/components/ExperimentList.tsx
import React, { useState, useEffect, useImperativeHandle, forwardRef } from 'react'; // Add useImperativeHandle, forwardRef
import { Link } from 'react-router-dom';
import './ExperimentList.css';

// Re-exporting interfaces for clarity if used elsewhere, but mainly for this file
interface ExperimentMetadata {
  id: string;
  status: string;
  created_at: string; // ISO string
  config_path: string;
  result_path?: string; // Optional for queued/running
}

interface CompletedExperiment {
  id: string;
  finished_at: string; // ISO string
  config: Record<string, any>; // JSON object
  metrics: Record<string, any>; // JSON object
  graph: Record<string, any>; // JSON object
}

interface StatusResponse {
  running: ExperimentMetadata | null;
  queue: ExperimentMetadata[];
  completed: CompletedExperiment[];
}

// Use forwardRef to allow parent to access fetchExperiments
const ExperimentList = forwardRef<
  { fetchExperiments: () => void }, // Define what parent can access
  {} // Define props (none currently, but could be added)
>((props, ref) => {
  const [runningExperiment, setRunningExperiment] = useState<ExperimentMetadata | null>(null);
  const [queuedExperiments, setQueuedExperiments] = useState<ExperimentMetadata[]>([]);
  const [pastExperiments, setPastExperiments] = useState<CompletedExperiment[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [removingId, setRemovingId] = useState<string | null>(null); // To show loading state for removal

  const fetchExperiments = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/status');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: StatusResponse = await response.json();
      setRunningExperiment(data.running);
      setQueuedExperiments(data.queue);
      setPastExperiments(data.completed);
    } catch (err) {
      console.error("Failed to fetch experiment status:", err);
      setError("Failed to load experiments. Please check the backend connection.");
    } finally {
      setLoading(false);
    }
  };

  // Expose fetchExperiments to parent via ref
  useImperativeHandle(ref, () => ({
    fetchExperiments,
  }));

  // Handle removing an experiment from the queue
  const handleRemoveFromQueue = async (id: string) => {
    if (!window.confirm(`Are you sure you want to remove experiment ${id} from the queue?`)) {
      return;
    }
    setRemovingId(id); // Set ID to show loading state for this specific item
    try {
      const response = await fetch(`http://localhost:8000/queue/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      // If successful, re-fetch all experiments to update the list
      await fetchExperiments();
      alert(`Experiment ${id} removed successfully.`);
    } catch (err) {
      console.error(`Failed to remove experiment ${id}:`, err);
      alert(`Failed to remove experiment ${id}: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setRemovingId(null); // Clear loading state
    }
  };


  useEffect(() => {
    fetchExperiments();
    // Refresh every 5 seconds to update status
    const interval = setInterval(fetchExperiments, 5000);
    return () => clearInterval(interval); // Cleanup on unmount
  }, []);

  if (loading) return <div className="loading">Loading experiments...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <div className="experiment-list-container">
      <section className="experiment-section">
        <h2>Currently Running Experiment</h2>
        {runningExperiment ? (
          <div className="experiment-card running-card">
            <h3>{runningExperiment.id}</h3>
            <p>Status: <span className="status-running">{runningExperiment.status}</span></p>
            <p>Created At: {new Date(runningExperiment.created_at).toLocaleString()}</p>
          </div>
        ) : (
          <p>No experiment currently running.</p>
        )}
      </section>

      <section className="experiment-section">
        <h2>Experiment Queue ({queuedExperiments.length})</h2>
        {queuedExperiments.length > 0 ? (
          <div className="experiment-grid">
            {queuedExperiments.map((exp) => (
              <div key={exp.id} className="experiment-card queued-card">
                <h3>{exp.id}</h3>
                <p>Status: <span className="status-queued">{exp.status}</span></p>
                <p>Created At: {new Date(exp.created_at).toLocaleString()}</p>
                <button
                  onClick={() => handleRemoveFromQueue(exp.id)}
                  disabled={removingId === exp.id} // Disable button while removing
                  className="remove-button"
                >
                  {removingId === exp.id ? 'Removing...' : 'Remove'}
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p>Queue is empty.</p>
        )}
      </section>

      <section className="experiment-section">
        <h2>Past Experiments ({pastExperiments.length})</h2>
        {pastExperiments.length > 0 ? (
          <div className="experiment-grid">
            {pastExperiments.map((exp) => (
              <Link to={`/experiments/${exp.id}`} key={exp.id} className="experiment-card past-card">
                <h3>{exp.id}</h3>
                <p>Status: <span className={`status-${exp.metrics.status === 'failed' ? 'failed' : 'completed'}`}>
                  {exp.metrics.status === 'failed' ? 'Failed' : 'Completed'}
                </span></p>
                <p>Finished At: {new Date(exp.finished_at).toLocaleString()}</p>
                {exp.metrics && exp.metrics.accuracy && <p>Accuracy: {(exp.metrics.accuracy * 100).toFixed(2)}%</p>}
                {exp.metrics && exp.metrics.error_message && <p className="error-message">Error: {exp.metrics.error_message}</p>}
              </Link>
            ))}
          </div>
        ) : (
          <p>No past experiments found.</p>
        )}
      </section>
    </div>
  );
});

export default ExperimentList;
