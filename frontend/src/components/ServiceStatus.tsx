import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Cog, Pause, Play } from 'lucide-react';
import QueryManager from './QueryManager';

type Health = { backend: boolean; crawler: boolean; knowledge_extraction: boolean };
type CrawlerStatus = 'idle' | 'running' | 'stopped' | 'error';

const services: { key: keyof Health; label: string }[] = [
  { key: 'crawler', label: 'Crawler' },
  { key: 'backend', label: 'Backend' },
  { key: 'knowledge_extraction', label: 'Knowledge Extractor' },
];

const ServiceStatus: React.FC = () => {
  const [health, setHealth] = useState<Health | null>(null);
  const [crawlerStatus, setCrawlerStatus] = useState<CrawlerStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const queryDialogRef = useRef<HTMLDialogElement>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      setHealth(await response.json());
    } catch (err) {
      console.error('Failed to fetch service health:', err);
      setHealth({ backend: false, crawler: false, knowledge_extraction: false });
    }

    try {
      const response = await fetch('http://localhost:8000/crawler/status');
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setCrawlerStatus(data.status);
    } catch (err) {
      setCrawlerStatus(null);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const toggleCrawler = async () => {
    const action = crawlerStatus === 'running' ? 'stop' : 'start';
    setBusy(true);
    try {
      const response = await fetch(`http://localhost:8000/crawler/${action}`, { method: 'POST' });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setCrawlerStatus(data.status);
    } catch (err) {
      console.error(`Failed to ${action} crawler:`, err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex items-center gap-2">
      {services.map(({ key, label }) => {
        const reachable = health?.[key] ?? false;
        return (
          <span
            key={key}
            className="inline-flex items-center gap-1.5 rounded-full border border-border bg-white px-2.5 py-0.5 text-xs font-medium text-foreground"
          >
            <span className={`size-1.5 rounded-full ${reachable ? 'bg-green-500' : 'bg-red-500'}`} />
            {label}
            {key === 'crawler' && reachable && (
              <>
                <button
                  type="button"
                  onClick={toggleCrawler}
                  disabled={busy}
                  aria-label={crawlerStatus === 'running' ? 'Stop crawler' : 'Start crawler'}
                  className="flex items-center text-muted-foreground hover:text-foreground disabled:opacity-50"
                >
                  {crawlerStatus === 'running' ? <Pause className="size-3" /> : <Play className="size-3" />}
                </button>
                <button
                  type="button"
                  onClick={() => queryDialogRef.current?.showModal()}
                  aria-label="Manage search queries"
                  className="flex items-center text-muted-foreground hover:text-foreground"
                >
                  <Cog className="size-3" />
                </button>
              </>
            )}
          </span>
        );
      })}
      <dialog
        ref={queryDialogRef}
        onClick={(e) => {
          if (e.target === queryDialogRef.current) queryDialogRef.current?.close();
        }}
        className="m-auto w-full max-w-md rounded-lg p-0 backdrop:bg-black/50"
      >
        <QueryManager />
      </dialog>
    </div>
  );
};

export default ServiceStatus;
