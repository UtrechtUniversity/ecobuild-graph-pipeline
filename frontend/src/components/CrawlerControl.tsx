// src/components/CrawlerControl.tsx
import React, { useState, useEffect, useCallback } from 'react';
import './CrawlerControl.css';

type CrawlerStatus = 'idle' | 'running' | 'stopped' | 'error';

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
    <div className="crawler-control">
      <h2>Paper Crawler</h2>
      <p>
        Status: <span className={`crawler-status crawler-status-${status ?? 'unknown'}`}>
          {status ?? 'unknown'}{status === 'error' && crawlError ? ` (${crawlError})` : ''}
        </span>
      </p>
      {error && <p className="crawler-error">{error}</p>}
      <button
        onClick={() => handleAction('start')}
        disabled={busy || status === 'running'}
        className="crawler-button"
      >
        Start
      </button>
      <button
        onClick={() => handleAction('stop')}
        disabled={busy || status !== 'running'}
        className="crawler-button"
      >
        Stop
      </button>
    </div>
  );
};

export default CrawlerControl;
