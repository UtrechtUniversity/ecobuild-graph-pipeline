import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';

interface SearchQuery {
  id: number;
  query: string;
  design_strategy: string | null;
  ecosystem_service: string | null;
}

interface QueryManagerProps {
  source: string;
  label: string;
}

const QueryManager: React.FC<QueryManagerProps> = ({ source, label }) => {
  const [queries, setQueries] = useState<SearchQuery[]>([]);
  const [newQuery, setNewQuery] = useState('');
  const [newDesignStrategy, setNewDesignStrategy] = useState('');
  const [newEcosystemService, setNewEcosystemService] = useState('');
  const [search, setSearch] = useState('');
  const [busy, setBusy] = useState(false);
  const [removingId, setRemovingId] = useState<number | null>(null);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState('');
  const [editDesignStrategy, setEditDesignStrategy] = useState('');
  const [editEcosystemService, setEditEcosystemService] = useState('');
  const [savingId, setSavingId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchQueries = useCallback(async () => {
    try {
      const response = await fetch(`http://localhost:8000/crawlers/${source}/queries`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      setQueries(await response.json());
      setError(null);
    } catch (err) {
      console.error('Failed to fetch search queries:', err);
      setError('Failed to reach the crawler.');
    }
  }, [source]);

  useEffect(() => {
    fetchQueries();
  }, [fetchQueries]);

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = newQuery.trim();
    if (!query) return;
    setBusy(true);
    try {
      const response = await fetch(`http://localhost:8000/crawlers/${source}/queries`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          design_strategy: newDesignStrategy.trim() || null,
          ecosystem_service: newEcosystemService.trim() || null,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      setNewQuery('');
      setNewDesignStrategy('');
      setNewEcosystemService('');
      await fetchQueries();
    } catch (err) {
      console.error('Failed to add query:', err);
      setError(err instanceof Error ? err.message : 'Failed to add query.');
    } finally {
      setBusy(false);
    }
  };

  const startEdit = (q: SearchQuery) => {
    setEditingId(q.id);
    setEditValue(q.query);
    setEditDesignStrategy(q.design_strategy ?? '');
    setEditEcosystemService(q.ecosystem_service ?? '');
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditValue('');
    setEditDesignStrategy('');
    setEditEcosystemService('');
  };

  const handleSaveEdit = async (id: number) => {
    const query = editValue.trim();
    if (!query) return;
    setSavingId(id);
    try {
      const response = await fetch(`http://localhost:8000/crawlers/${source}/queries/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          design_strategy: editDesignStrategy.trim() || null,
          ecosystem_service: editEcosystemService.trim() || null,
        }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      cancelEdit();
      await fetchQueries();
    } catch (err) {
      console.error('Failed to update query:', err);
      setError(err instanceof Error ? err.message : 'Failed to update query.');
    } finally {
      setSavingId(null);
    }
  };

  const handleRemove = async (id: number) => {
    setRemovingId(id);
    try {
      const response = await fetch(`http://localhost:8000/crawlers/${source}/queries/${id}`, { method: 'DELETE' });
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
    <Card className="flex h-full flex-col overflow-hidden">
      <CardHeader>
        <CardTitle>{label} Search Queries</CardTitle>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 flex-col gap-4">
        {error && <p className="text-sm text-destructive">{error}</p>}
        {queries.length > 0 && (
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search queries…"
            className="h-9 rounded-md border border-input bg-card px-3 text-sm shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          />
        )}
        <ul className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto">
          {queries
            .filter((q) => q.query.toLowerCase().includes(search.toLowerCase()))
            .map((q) =>
              editingId === q.id ? (
                <li key={q.id} className="flex flex-col gap-2 rounded-md border border-input px-3 py-2 text-sm">
                  <input
                    type="text"
                    autoFocus
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleSaveEdit(q.id);
                      if (e.key === 'Escape') cancelEdit();
                    }}
                    className="h-8 rounded-md border border-input bg-card px-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                  />
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={editDesignStrategy}
                      onChange={(e) => setEditDesignStrategy(e.target.value)}
                      placeholder="Design strategy (DS)"
                      className="h-8 flex-1 rounded-md border border-input bg-card px-2 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                    />
                    <input
                      type="text"
                      value={editEcosystemService}
                      onChange={(e) => setEditEcosystemService(e.target.value)}
                      placeholder="Ecosystem service (ES)"
                      className="h-8 flex-1 rounded-md border border-input bg-card px-2 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                    />
                  </div>
                  <div className="flex justify-end gap-2">
                    <Button size="sm" onClick={() => handleSaveEdit(q.id)} disabled={savingId === q.id || !editValue.trim()}>
                      {savingId === q.id ? 'Saving…' : 'Save'}
                    </Button>
                    <Button variant="outline" size="sm" onClick={cancelEdit} disabled={savingId === q.id}>
                      Cancel
                    </Button>
                  </div>
                </li>
              ) : (
                <li key={q.id} className="flex items-center justify-between gap-2 rounded-md border border-input px-3 py-2 text-sm">
                  <div className="flex flex-col gap-0.5">
                    <span>{q.query}</span>
                    {(q.design_strategy || q.ecosystem_service) && (
                      <span className="text-xs text-muted-foreground">
                        {q.design_strategy && <>DS: {q.design_strategy}</>}
                        {q.design_strategy && q.ecosystem_service && ' · '}
                        {q.ecosystem_service && <>ES: {q.ecosystem_service}</>}
                      </span>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={() => startEdit(q)}>
                      Edit
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => handleRemove(q.id)}
                      disabled={removingId === q.id}
                    >
                      {removingId === q.id ? 'Removing…' : 'Remove'}
                    </Button>
                  </div>
                </li>
              )
            )}
          {queries.length === 0 && (
            <p className="text-sm text-muted-foreground">No search queries configured.</p>
          )}
        </ul>
        <form onSubmit={handleAdd} className="flex flex-col gap-2">
          <input
            type="text"
            value={newQuery}
            onChange={(e) => setNewQuery(e.target.value)}
            placeholder="Add a search query…"
            className="h-9 rounded-md border border-input bg-card px-3 text-sm shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          />
          <div className="flex gap-2">
            <input
              type="text"
              value={newDesignStrategy}
              onChange={(e) => setNewDesignStrategy(e.target.value)}
              placeholder="Design strategy (DS, optional)"
              className="h-9 flex-1 rounded-md border border-input bg-card px-3 text-xs shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            />
            <input
              type="text"
              value={newEcosystemService}
              onChange={(e) => setNewEcosystemService(e.target.value)}
              placeholder="Ecosystem service (ES, optional)"
              className="h-9 flex-1 rounded-md border border-input bg-card px-3 text-xs shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
            />
            <Button type="submit" disabled={busy || !newQuery.trim()}>
              Add
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
};

export default QueryManager;
