// src/components/ExperimentList.tsx
import React, { useState, useEffect, useImperativeHandle, forwardRef } from 'react';
import { Link } from 'react-router-dom';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

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
  { fetchExperiments: () => void },
  {}
>((props, ref) => {
  const [runningExperiment, setRunningExperiment] = useState<ExperimentMetadata | null>(null);
  const [queuedExperiments, setQueuedExperiments] = useState<ExperimentMetadata[]>([]);
  const [pastExperiments, setPastExperiments] = useState<CompletedExperiment[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [removingId, setRemovingId] = useState<string | null>(null);

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

  useImperativeHandle(ref, () => ({
    fetchExperiments,
  }));

  const handleRemoveFromQueue = async (id: string) => {
    if (!window.confirm(`Are you sure you want to remove experiment ${id} from the queue?`)) {
      return;
    }
    setRemovingId(id);
    try {
      const response = await fetch(`http://localhost:8000/queue/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      await fetchExperiments();
      alert(`Experiment ${id} removed successfully.`);
    } catch (err) {
      console.error(`Failed to remove experiment ${id}:`, err);
      alert(`Failed to remove experiment ${id}: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setRemovingId(null);
    }
  };

  useEffect(() => {
    fetchExperiments();
    const interval = setInterval(fetchExperiments, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) return <p className="py-8 text-center text-muted-foreground">Loading experiments…</p>;
  if (error) return <p className="py-8 text-center text-destructive">{error}</p>;

  return (
    <div className="flex flex-col gap-8">
      <Card>
        <CardHeader>
          <CardTitle>Currently Running</CardTitle>
        </CardHeader>
        <CardContent>
          {runningExperiment ? (
            <Card className="border-primary/40 bg-accent">
              <CardContent className="pt-5">
                <h3 className="font-medium">{runningExperiment.id}</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Status: <Badge variant="success">{runningExperiment.status}</Badge>
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Created: {new Date(runningExperiment.created_at).toLocaleString()}
                </p>
              </CardContent>
            </Card>
          ) : (
            <p className="text-sm text-muted-foreground">No experiment currently running.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Experiment Queue ({queuedExperiments.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {queuedExperiments.length > 0 ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {queuedExperiments.map((exp) => (
                <Card key={exp.id}>
                  <CardContent className="flex flex-col gap-1 pt-5">
                    <h3 className="font-medium">{exp.id}</h3>
                    <p className="text-sm text-muted-foreground">
                      Status: <Badge variant="warning">{exp.status}</Badge>
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Created: {new Date(exp.created_at).toLocaleString()}
                    </p>
                    <Button
                      variant="destructive"
                      size="sm"
                      className="mt-2 self-start"
                      onClick={() => handleRemoveFromQueue(exp.id)}
                      disabled={removingId === exp.id}
                    >
                      {removingId === exp.id ? 'Removing…' : 'Remove'}
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Queue is empty.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Past Experiments ({pastExperiments.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {pastExperiments.length > 0 ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {pastExperiments.map((exp) => {
                const failed = exp.metrics.status === 'failed';
                return (
                  <Link to={`/experiments/${exp.id}`} key={exp.id}>
                    <Card className="h-full transition-shadow hover:shadow-md">
                      <CardContent className="flex flex-col gap-1 pt-5">
                        <h3 className="font-medium">{exp.id}</h3>
                        <p className="text-sm text-muted-foreground">
                          Status: <Badge variant={failed ? 'destructive' : 'success'}>{failed ? 'Failed' : 'Completed'}</Badge>
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Finished: {new Date(exp.finished_at).toLocaleString()}
                        </p>
                        {exp.metrics?.accuracy != null && (
                          <p className="text-sm text-muted-foreground">
                            Accuracy: {(exp.metrics.accuracy * 100).toFixed(2)}%
                          </p>
                        )}
                        {exp.metrics?.error_message && (
                          <p className="text-sm font-medium text-destructive">
                            Error: {exp.metrics.error_message}
                          </p>
                        )}
                      </CardContent>
                    </Card>
                  </Link>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No past experiments found.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
});

export default ExperimentList;
