import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import './ExperimentDetail.css'; // We'll create this

interface CompletedExperimentDetail {
  id: string;
  finished_at: string;
  config: Record<string, any>;
  metrics: Record<string, any>;
  graph: Record<string, any>;
}

const ExperimentDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [experiment, setExperiment] = useState<CompletedExperimentDetail | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchExperimentDetail = async () => {
      if (!id) return;
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:8000/experiments/${id}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: CompletedExperimentDetail = await response.json();
        setExperiment(data);
      } catch (err) {
        console.error(`Failed to fetch experiment ${id} details:`, err);
        setError(`Failed to load details for experiment ${id}.`);
      } finally {
        setLoading(false);
      }
    };

    fetchExperimentDetail();
  }, [id]);

  if (loading) return <div className="loading">Loading experiment details...</div>;
  if (error) return <div className="error">{error}</div>;
  if (!experiment) return <div className="error">Experiment not found.</div>;

  return (
    <div className="experiment-detail-container">
      <h2>Experiment Details: {experiment.id}</h2>
      <p>Finished At: {new Date(experiment.finished_at).toLocaleString()}</p>

      <div className="detail-panels">
        <div className="detail-panel metrics-panel">
          <h3>Metrics</h3>
          <pre>{JSON.stringify(experiment.metrics, null, 2)}</pre>
        </div>
        <div className="detail-panel config-panel">
          <h3>Configuration</h3>
          <pre>{JSON.stringify(experiment.config, null, 2)}</pre>
        </div>
        <div className="detail-panel graph-panel">
          <h3>Neo4j Graph (JSON)</h3>
          <pre>{JSON.stringify(experiment.graph, null, 2)}</pre>
        </div>
      </div>
    </div>
  );
};

export default ExperimentDetail;
