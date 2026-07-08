import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

type ExtractionStatus = 'pending' | 'downloading' | 'extracting' | 'done' | 'failed';

interface Paper {
  id: number;
  title: string;
  authors: string[];
  query: string | null;
  open_access: boolean | null;
  pdf_url: string | null;
  extraction_status: ExtractionStatus;
  extraction_error: string | null;
}

const statusVariant: Record<ExtractionStatus, 'secondary' | 'warning' | 'success' | 'destructive'> = {
  pending: 'secondary',
  downloading: 'warning',
  extracting: 'warning',
  done: 'success',
  failed: 'destructive',
};

const PaperList: React.FC = () => {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [results, setResults] = useState<Record<number, Record<string, unknown>>>({});
  const [resultsError, setResultsError] = useState<string | null>(null);

  const fetchPapers = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/papers');
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      setPapers(await response.json());
      setError(null);
    } catch (err) {
      console.error('Failed to fetch papers:', err);
      setError('Failed to reach the backend.');
    }
  }, []);

  useEffect(() => {
    fetchPapers();
    const interval = setInterval(fetchPapers, 5000);
    return () => clearInterval(interval);
  }, [fetchPapers]);

  const toggle = (id: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    setSelected((prev) => (prev.size === papers.length ? new Set() : new Set(papers.map((p) => p.id))));
  };

  const toggleExpanded = async (paper: Paper) => {
    if (expandedId === paper.id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(paper.id);
    setResultsError(null);
    if (results[paper.id]) return;
    try {
      const response = await fetch(`http://localhost:8000/papers/${paper.id}/results`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setResults((prev) => ({ ...prev, [paper.id]: data }));
    } catch (err) {
      console.error(`Failed to fetch results for paper ${paper.id}:`, err);
      setResultsError('Failed to load extraction results.');
    }
  };

  const handleExtract = async () => {
    setBusy(true);
    try {
      const response = await fetch('http://localhost:8000/papers/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paper_ids: Array.from(selected) }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      setSelected(new Set());
      await fetchPapers();
    } catch (err) {
      console.error('Failed to start extraction:', err);
      setError('Failed to start extraction.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Papers ({papers.length})</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {error && <p className="text-sm text-destructive">{error}</p>}
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              checked={papers.length > 0 && selected.size === papers.length}
              onChange={toggleAll}
            />
            Select all
          </label>
          <Button onClick={handleExtract} disabled={busy || selected.size === 0}>
            {busy ? 'Starting…' : `Extract selected (${selected.size})`}
          </Button>
        </div>
        <ul className="flex flex-col gap-2">
          {papers.map((paper) => {
            const extracted = paper.extraction_status === 'done';
            return (
              <li key={paper.id} className="rounded-md border border-input px-3 py-2 text-sm">
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    className="mt-1"
                    checked={selected.has(paper.id)}
                    onChange={() => toggle(paper.id)}
                  />
                  <div className="flex flex-1 flex-col gap-1">
                    {extracted ? (
                      <button
                        type="button"
                        onClick={() => toggleExpanded(paper)}
                        className="text-left font-medium underline-offset-2 hover:underline"
                      >
                        {paper.title}
                      </button>
                    ) : (
                      <span className="font-medium">{paper.title}</span>
                    )}
                    <span className="text-xs text-muted-foreground">
                      {paper.authors.join(', ') || 'Unknown authors'}
                      {paper.query && ` · found via "${paper.query}"`}
                    </span>
                    {paper.extraction_status === 'failed' && paper.extraction_error && (
                      <span className="text-xs text-destructive">{paper.extraction_error}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={paper.open_access ? 'success' : 'outline'}>
                      {paper.open_access ? 'Open access' : 'Closed access'}
                    </Badge>
                    <Badge variant={statusVariant[paper.extraction_status]} className="capitalize">
                      {paper.extraction_status}
                    </Badge>
                  </div>
                </div>
                {extracted && expandedId === paper.id && (
                  <div className="mt-3 border-t border-input pt-3">
                    {resultsError ? (
                      <p className="text-sm text-destructive">{resultsError}</p>
                    ) : (
                      <pre className="max-h-100 overflow-auto rounded-md bg-muted p-3 font-mono text-xs whitespace-pre-wrap break-all">
                        {results[paper.id] ? JSON.stringify(results[paper.id], null, 2) : 'Loading…'}
                      </pre>
                    )}
                  </div>
                )}
              </li>
            );
          })}
          {papers.length === 0 && (
            <p className="text-sm text-muted-foreground">No papers found yet.</p>
          )}
        </ul>
      </CardContent>
    </Card>
  );
};

export default PaperList;
