import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';

interface SearchQuery {
  id: number;
  query: string;
}

const QueryManager: React.FC = () => {
  const [queries, setQueries] = useState<SearchQuery[]>([]);
  const [newQuery, setNewQuery] = useState('');
  const [busy, setBusy] = useState(false);
  const [removingId, setRemovingId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchQueries = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/crawler/queries');
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      setQueries(await response.json());
      setError(null);
    } catch (err) {
      console.error('Failed to fetch search queries:', err);
      setError('Failed to reach the crawler.');
    }
  }, []);

  useEffect(() => {
    fetchQueries();
  }, [fetchQueries]);

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = newQuery.trim();
    if (!query) return;
    setBusy(true);
    try {
      const response = await fetch('http://localhost:8000/crawler/queries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      setNewQuery('');
      await fetchQueries();
    } catch (err) {
      console.error('Failed to add query:', err);
      setError(err instanceof Error ? err.message : 'Failed to add query.');
    } finally {
      setBusy(false);
    }
  };

  const handleRemove = async (id: number) => {
    setRemovingId(id);
    try {
      const response = await fetch(`http://localhost:8000/crawler/queries/${id}`, { method: 'DELETE' });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchQueries();
    } catch (err) {
      console.error('Failed to remove query:', err);
      setError('Failed to remove query.');
    } finally {
      setRemovingId(null);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Search Queries</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {error && <p className="text-sm text-destructive">{error}</p>}
        <ul className="flex flex-col gap-2">
          {queries.map((q) => (
            <li key={q.id} className="flex items-center justify-between gap-2 rounded-md border border-input px-3 py-2 text-sm">
              <span>{q.query}</span>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => handleRemove(q.id)}
                disabled={removingId === q.id}
              >
                {removingId === q.id ? 'Removing…' : 'Remove'}
              </Button>
            </li>
          ))}
          {queries.length === 0 && (
            <p className="text-sm text-muted-foreground">No search queries configured.</p>
          )}
        </ul>
        <form onSubmit={handleAdd} className="flex gap-2">
          <input
            type="text"
            value={newQuery}
            onChange={(e) => setNewQuery(e.target.value)}
            placeholder="Add a search query…"
            className="h-9 flex-1 rounded-md border border-input bg-card px-3 text-sm shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          />
          <Button type="submit" disabled={busy || !newQuery.trim()}>
            Add
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default QueryManager;
