import React, { useState, useEffect } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';

interface CompletedExperimentDetail {
  id: string;
  finished_at: string;
  config: Record<string, any>;
  metrics: Record<string, any>;
  graph: Record<string, any>;
}

const panels: { key: keyof CompletedExperimentDetail; title: string; accent: string }[] = [
  { key: 'metrics', title: 'Metrics', accent: 'border-l-warning' },
  { key: 'config', title: 'Configuration', accent: 'border-l-primary' },
  { key: 'graph', title: 'Neo4j Graph (JSON)', accent: 'border-l-success' },
];

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

  if (loading) return <p className="py-8 text-center text-muted-foreground">Loading experiment details…</p>;
  if (error) return <p className="py-8 text-center text-destructive">{error}</p>;
  if (!experiment) return <p className="py-8 text-center text-destructive">Experiment not found.</p>;

  return (
    <div className="flex flex-col gap-6">
      <Link to="/" className="inline-flex w-fit items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="size-4" /> Back to dashboard
      </Link>

      <div>
        <h2 className="font-serif text-2xl font-semibold">{experiment.id}</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Finished: {new Date(experiment.finished_at).toLocaleString()}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {panels.map(({ key, title, accent }) => (
          <Card key={key} className={`border-l-4 ${accent}`}>
            <CardHeader>
              <CardTitle className="text-base">{title}</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="max-h-125 overflow-auto rounded-md bg-muted p-3 font-mono text-xs whitespace-pre-wrap break-all">
                {JSON.stringify(experiment[key], null, 2)}
              </pre>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default ExperimentDetail;
