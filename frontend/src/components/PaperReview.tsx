import React, { useState, useEffect } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';

const PaperReview: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchResults = async () => {
      if (!id) return;
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:8000/papers/${id}/results`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        setResults(await response.json());
        setError(null);
      } catch (err) {
        console.error(`Failed to fetch results for paper ${id}:`, err);
        setError(`Failed to load extraction results for paper ${id}.`);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [id]);

  return (
    <div className="flex flex-col gap-6">
      <Link to="/" className="inline-flex w-fit items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="size-4" /> Back to papers
      </Link>

      <h2 className="font-serif text-2xl font-semibold">Paper #{id} — extraction results</h2>

      {loading && <p className="text-sm text-muted-foreground">Loading results…</p>}
      {error && <p className="text-sm text-destructive">{error}</p>}

      {results && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Results</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="max-h-125 overflow-auto rounded-md bg-muted p-3 font-mono text-xs whitespace-pre-wrap break-all">
              {JSON.stringify(results, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default PaperReview;
