import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

type CrawlerStatus = 'idle' | 'running' | 'stopped' | 'error';

const statusVariant: Record<string, 'default' | 'secondary' | 'destructive' | 'success' | 'warning'> = {
  running: 'success',
  idle: 'secondary',
  stopped: 'warning',
  error: 'destructive',
};

const CrawlerControl: React.FC = () => {
  const [status, setStatus] = useState<CrawlerStatus | null>(null);
  const [crawlError, setCrawlError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/crawler/status');
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setStatus(data.status);
      setCrawlError(data.error ?? null);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch crawler status:', err);
      setError('Failed to reach the crawler.');
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleAction = async (action: 'start' | 'stop') => {
    setBusy(true);
    try {
      const response = await fetch(`http://localhost:8000/crawler/${action}`, { method: 'POST' });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setStatus(data.status);
      setCrawlError(data.error ?? null);
      setError(null);
    } catch (err) {
      console.error(`Failed to ${action} crawler:`, err);
      setError(`Failed to ${action} the crawler.`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Paper Crawler</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Status:</span>
          <Badge variant={statusVariant[status ?? ''] ?? 'outline'} className="capitalize">
            {status ?? 'unknown'}
          </Badge>
          {status === 'error' && crawlError && (
            <span className="text-destructive">({crawlError})</span>
          )}
        </div>
        {error && <p className="text-sm text-destructive">{error}</p>}
        <div className="flex gap-2">
          <Button onClick={() => handleAction('start')} disabled={busy || status === 'running'}>
            Start
          </Button>
          <Button
            variant="outline"
            onClick={() => handleAction('stop')}
            disabled={busy || status !== 'running'}
          >
            Stop
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default CrawlerControl;
