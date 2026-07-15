import { API_BASE } from '../config';
import React, { useCallback, useEffect, useState } from 'react';
import { Cog, Pause, Play } from 'lucide-react';
import QueryManager from './QueryManager';

const CRAWLERS = [
  { key: 'semantic_scholar', label: 'Semantic Scholar' },
  { key: 'scopus', label: 'Scopus' },
] as const;

type CrawlerKey = (typeof CRAWLERS)[number]['key'];
type Health = { backend: boolean; knowledge_extraction: boolean } & Record<CrawlerKey, boolean>;
type CrawlerStatus = 'idle' | 'running' | 'stopped' | 'error';

const services: { key: keyof Health; label: string }[] = [
  ...CRAWLERS,
  { key: 'backend', label: 'Backend' },
  { key: 'knowledge_extraction', label: 'Knowledge Extractor' },
];

const ServiceStatus: React.FC = () => {
  const [health, setHealth] = useState<Health | null>(null);
  const [crawlerStatus, setCrawlerStatus] = useState<Record<CrawlerKey, CrawlerStatus | null>>(
    Object.fromEntries(CRAWLERS.map((c) => [c.key, null])) as Record<CrawlerKey, CrawlerStatus | null>,
  );
  const [busyKey, setBusyKey] = useState<CrawlerKey | null>(null);
  const [queryDialogSource, setQueryDialogSource] = useState<CrawlerKey | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      setHealth(await response.json());
    } catch (err) {
      console.error('Failed to fetch service health:', err);
      setHealth(null);
    }

    const statuses = await Promise.all(
      CRAWLERS.map(async ({ key }) => {
        try {
          const response = await fetch(`${API_BASE}/crawlers/${key}/status`);
          if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
          const data = await response.json();
          return [key, data.status as CrawlerStatus] as const;
        } catch {
          return [key, null] as const;
        }
      }),
    );
    setCrawlerStatus(Object.fromEntries(statuses) as Record<CrawlerKey, CrawlerStatus | null>);
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  useEffect(() => {
    if (!queryDialogSource) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setQueryDialogSource(null);
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [queryDialogSource]);

  const toggleCrawler = async (key: CrawlerKey) => {
    const action = crawlerStatus[key] === 'running' ? 'stop' : 'start';
    setBusyKey(key);
    try {
      const response = await fetch(`${API_BASE}/crawlers/${key}/${action}`, { method: 'POST' });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setCrawlerStatus((prev) => ({ ...prev, [key]: data.status }));
    } catch (err) {
      console.error(`Failed to ${action} crawler '${key}':`, err);
    } finally {
      setBusyKey(null);
    }
  };

  const dialogCrawler = CRAWLERS.find((c) => c.key === queryDialogSource);

  return (
    <div className="flex items-center gap-2">
      {services.map(({ key, label }) => {
        const reachable = health?.[key] ?? false;
        const isCrawler = CRAWLERS.some((c) => c.key === key);
        return (
          <span
            key={key}
            className="inline-flex items-center gap-1.5 rounded-full border border-border bg-white px-2.5 py-0.5 text-xs font-medium text-foreground"
          >
            <span className={`size-1.5 rounded-full ${reachable ? 'bg-green-500' : 'bg-red-500'}`} />
            {label}
            {isCrawler && reachable && (
              <>
                <button
                  type="button"
                  onClick={() => toggleCrawler(key as CrawlerKey)}
                  disabled={busyKey === key}
                  aria-label={crawlerStatus[key as CrawlerKey] === 'running' ? `Stop ${label}` : `Start ${label}`}
                  className="flex items-center text-muted-foreground hover:text-foreground disabled:opacity-50"
                >
                  {crawlerStatus[key as CrawlerKey] === 'running' ? <Pause className="size-3" /> : <Play className="size-3" />}
                </button>
                <button
                  type="button"
                  onClick={() => setQueryDialogSource(key as CrawlerKey)}
                  aria-label={`Manage ${label} search queries`}
                  className="flex items-center text-muted-foreground hover:text-foreground"
                >
                  <Cog className="size-3" />
                </button>
              </>
            )}
          </span>
        );
      })}
      {dialogCrawler && (
        <div
          role="dialog"
          aria-modal="true"
          onClick={(e) => {
            if (e.target === e.currentTarget) setQueryDialogSource(null);
          }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
        >
          <div style={{ height: '70vh', maxWidth: '640px' }} className="flex w-full flex-col rounded-lg">
            <QueryManager source={dialogCrawler.key} label={dialogCrawler.label} />
          </div>
        </div>
      )}
    </div>
  );
};

export default ServiceStatus;
