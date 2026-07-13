import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import { Loader2, CircleAlert, Play, RotateCcw, Upload } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

type ExtractionStatus = 'pending' | 'downloading' | 'downloaded' | 'extracting' | 'done' | 'failed';

const spinningStatuses: ExtractionStatus[] = ['downloading', 'extracting'];

interface Paper {
  id: number;
  title: string;
  authors: string[];
  query: string | null;
  open_access: boolean | null;
  pdf_url: string | null;
  ss_id: string;
  url: string | null;
  doi: string | null;
  abstract: string | null;
  relevance_checked: boolean | null;
  relevant: boolean | null;
  created_at: string | null;
  extraction_status: ExtractionStatus;
  extraction_error: string | null;
  extraction_started_at: string | null;
}

function formatElapsed(startedAt: string): string {
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000));
  const minutes = Math.floor(seconds / 60);
  const elapsed = minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
  const started = new Date(startedAt).toLocaleTimeString();
  return `Started ${started} · ${elapsed} ago`;
}

function highlightQueryMatches(query: string, title: string): React.ReactNode {
  const titleLower = title.toLowerCase();
  const parts = query.split(/("[^"]*")/g);
  return parts.map((part, i) => {
    const phrase = part.replace(/^"|"$/g, '');
    if (part.startsWith('"') && phrase && titleLower.includes(phrase.toLowerCase())) {
      return <strong key={i}>{part}</strong>;
    }
    return part;
  });
}

const statusVariant: Record<ExtractionStatus, 'secondary' | 'warning' | 'success' | 'destructive'> = {
  pending: 'secondary',
  downloading: 'warning',
  downloaded: 'warning',
  extracting: 'warning',
  done: 'success',
  failed: 'destructive',
};

interface ResultTag {
  id: number;
  tag_type: string;
  value: string | null;
}

function humanizeTagType(tagType: string): string {
  return tagType.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

// Status pill with a hover tooltip showing what the run is doing / its outcome.
// Clicking pins it open so a long results list can be scrolled without the
// mouse having to stay put.
function StatusPill({ paper }: { paper: Paper }) {
  const [hovered, setHovered] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [tags, setTags] = useState<ResultTag[] | null>(null);
  const [tagsFailed, setTagsFailed] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const showPanel = hovered || pinned;

  useEffect(() => {
    if (!pinned) return;
    const onOutsideClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setPinned(false);
    };
    document.addEventListener('mousedown', onOutsideClick);
    return () => document.removeEventListener('mousedown', onOutsideClick);
  }, [pinned]);

  useEffect(() => {
    if (!showPanel || paper.extraction_status !== 'done' || tags !== null) return;
    fetch(`http://localhost:8000/papers/${paper.id}/results`)
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then((data) => setTags(data.tags))
      .catch(() => setTagsFailed(true));
  }, [showPanel, paper.extraction_status, paper.id, tags]);

  if (paper.extraction_status === 'pending') {
    return <Badge variant={statusVariant[paper.extraction_status]} className="capitalize">{paper.extraction_status}</Badge>;
  }

  return (
    <div
      ref={containerRef}
      className="relative"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <Badge
        variant={statusVariant[paper.extraction_status]}
        className="capitalize cursor-pointer"
        onClick={() => setPinned((p) => !p)}
      >
        {paper.extraction_status === 'extracting' && <Loader2 className="size-3 animate-spin" />}
        {paper.extraction_status === 'failed' && <CircleAlert className="size-3" />}
        {paper.extraction_status === 'failed' ? 'Error' : paper.extraction_status}
      </Badge>
      {showPanel && (
        <div className="absolute right-0 top-full z-20 mt-1 w-80 max-h-72 overflow-y-auto rounded-md border border-border bg-popover p-3 text-xs text-popover-foreground shadow-lg">
          {paper.extraction_status === 'failed' && (
            <p className="whitespace-pre-wrap">{paper.extraction_error ?? 'Unknown error.'}</p>
          )}
          {(paper.extraction_status === 'downloading' || paper.extraction_status === 'downloaded' || paper.extraction_status === 'extracting') && (
            <p className="flex flex-col gap-1">
              <span>
                {paper.extraction_status === 'downloading' && 'Downloading PDF…'}
                {paper.extraction_status === 'downloaded' && 'PDF downloaded, queued for extraction…'}
                {paper.extraction_status === 'extracting' && 'Running extraction (LLM + embedding calls over the full paper)…'}
              </span>
              {paper.extraction_started_at && (
                <span className="text-muted-foreground">{formatElapsed(paper.extraction_started_at)}</span>
              )}
            </p>
          )}
          {paper.extraction_status === 'done' && (
            tagsFailed ? (
              <p className="text-destructive">Failed to load results.</p>
            ) : tags === null ? (
              <p className="text-muted-foreground">Loading…</p>
            ) : tags.length === 0 ? (
              <p className="text-muted-foreground">No tags extracted.</p>
            ) : (
              <ul className="flex flex-col gap-1">
                {tags.map((tag) => (
                  <li key={tag.id} className="flex justify-between gap-3 border-b border-border/50 pb-1 last:border-0">
                    <span className="shrink-0 text-muted-foreground">{humanizeTagType(tag.tag_type)}</span>
                    <span className="text-right">{tag.value ?? '—'}</span>
                  </li>
                ))}
              </ul>
            )
          )}
        </div>
      )}
    </div>
  );
}

const PaperList: React.FC = () => {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadingId, setUploadingId] = useState<number | null>(null);
  const [dragOverId, setDragOverId] = useState<number | null>(null);
  // ponytail: extraction failures arrive as 200s (status polled from /papers, not
  // thrown), so there's no network error to catch — this logs them once each so
  // they still show up in devtools like a real error would.
  const loggedFailures = useRef<Set<number>>(new Set());

  const fetchPapers = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/papers');
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data: Paper[] = await response.json();
      setPapers(data);
      setError(null);
      for (const paper of data) {
        if (paper.extraction_status === 'failed' && !loggedFailures.current.has(paper.id)) {
          loggedFailures.current.add(paper.id);
          console.error(`Extraction failed for paper ${paper.id} (${paper.title}): ${paper.extraction_error}`);
        }
      }
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

  const startExtraction = async (paperIds: number[]) => {
    setBusy(true);
    try {
      const response = await fetch('http://localhost:8000/papers/extract', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paper_ids: paperIds }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchPapers();
    } catch (err) {
      console.error('Failed to start extraction:', err);
      setError('Failed to start extraction.');
    } finally {
      setBusy(false);
    }
  };

  const handleExtract = async () => {
    await startExtraction(Array.from(selected));
    setSelected(new Set());
  };

  // Manual fallback for papers automated download can't reach (bot-walled
  // publishers, etc.) — drops the PDF into the backend's watched folder,
  // which picks it up and runs it through the normal pipeline within a few
  // seconds; the existing 5s poll then just picks up the status change.
  const uploadPdf = async (paperId: number, file: File) => {
    setUploadingId(paperId);
    try {
      const body = new FormData();
      body.append('file', file);
      const response = await fetch(`http://localhost:8000/papers/${paperId}/upload`, { method: 'POST', body });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchPapers();
    } catch (err) {
      console.error('Failed to upload PDF:', err);
      setError('Failed to upload PDF.');
    } finally {
      setUploadingId(null);
    }
  };

  const dropZoneHandlers = (paperId: number) => ({
    onDragOver: (e: React.DragEvent) => { e.preventDefault(); setDragOverId(paperId); },
    onDragLeave: () => setDragOverId((id) => (id === paperId ? null : id)),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      setDragOverId(null);
      const file = e.dataTransfer.files[0];
      if (file) uploadPdf(paperId, file);
    },
  });

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
                  <div className="mt-0.5 flex items-center gap-1.5">
                    <input
                      type="checkbox"
                      checked={selected.has(paper.id)}
                      onChange={() => toggle(paper.id)}
                    />
                    {paper.extraction_status !== 'downloading' && paper.extraction_status !== 'downloaded' && paper.extraction_status !== 'extracting' && (
                      <Button
                        size="icon"
                        variant="outline"
                        className="size-6"
                        disabled={busy}
                        title={paper.extraction_status === 'pending' ? 'Start extraction' : 'Restart extraction'}
                        onClick={() => startExtraction([paper.id])}
                      >
                        {paper.extraction_status === 'pending' ? <Play className="size-3.5" /> : <RotateCcw className="size-3.5" />}
                      </Button>
                    )}
                  </div>
                  <div className="flex flex-1 flex-col gap-1">
                    <span className="font-medium">
                      <span className="text-muted-foreground">#{paper.id}</span>{' '}
                      {paper.url ? (
                        <a href={paper.url} target="_blank" rel="noreferrer" className="underline hover:text-primary">
                          {paper.title}
                        </a>
                      ) : (
                        paper.title
                      )}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {paper.authors.join(', ') || 'Unknown authors'}
                      {paper.query && (
                        <>
                          {' · found via "'}
                          {highlightQueryMatches(paper.query, paper.title)}
                          {'"'}
                        </>
                      )}
                    </span>
                    {paper.extraction_status === 'failed' && paper.extraction_error && (
                      <span className="flex items-center gap-1 rounded border border-destructive/20 bg-destructive/10 px-2 py-1 text-xs text-destructive">
                        <CircleAlert className="size-3 shrink-0" />
                        {paper.extraction_error}
                      </span>
                    )}
                    {(paper.extraction_status === 'pending' || paper.extraction_status === 'failed') && (
                      <label
                        {...dropZoneHandlers(paper.id)}
                        className={`flex cursor-pointer items-center gap-1.5 rounded border border-dashed px-2 py-1 text-xs text-muted-foreground hover:text-foreground ${
                          dragOverId === paper.id ? 'border-primary bg-primary/5 text-foreground' : 'border-input'
                        }`}
                      >
                        <Upload className="size-3 shrink-0" />
                        {uploadingId === paper.id ? 'Uploading…' : 'Drop or choose a PDF to upload manually'}
                        <input
                          type="file"
                          accept="application/pdf"
                          className="hidden"
                          disabled={uploadingId === paper.id}
                          onChange={(e) => { const file = e.target.files?.[0]; if (file) uploadPdf(paper.id, file); e.target.value = ''; }}
                        />
                      </label>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={paper.open_access ? 'success' : 'outline'}>
                      {paper.open_access ? 'Open access' : 'Closed access'}
                    </Badge>
                    <StatusPill paper={paper} />
                    {extracted && (
                      <Button asChild size="sm" variant="outline">
                        <Link to={`/papers/${paper.id}/review`}>Investigate results</Link>
                      </Button>
                    )}
                  </div>
                </div>
                <details className="mt-2">
                  <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
                    More info
                  </summary>
                  <div className="mt-2 flex flex-col gap-1 border-t border-input pt-2 text-xs text-muted-foreground">
                    {paper.abstract && (
                      <p><span className="font-medium text-foreground">Abstract: </span>{paper.abstract}</p>
                    )}
                    {paper.doi && (
                      <p>
                        <span className="font-medium text-foreground">DOI: </span>
                        <a href={`https://doi.org/${paper.doi}`} target="_blank" rel="noreferrer" className="underline">{paper.doi}</a>
                      </p>
                    )}
                    {paper.url && (
                      <p>
                        <span className="font-medium text-foreground">Source: </span>
                        <a href={paper.url} target="_blank" rel="noreferrer" className="underline">{paper.url}</a>
                      </p>
                    )}
                    <p><span className="font-medium text-foreground">Semantic Scholar ID: </span>{paper.ss_id}</p>
                    <p>
                      <span className="font-medium text-foreground">Relevance: </span>
                      {paper.relevance_checked ? (paper.relevant ? 'Checked — relevant' : 'Checked — not relevant') : 'Not checked'}
                    </p>
                    {paper.created_at && (
                      <p><span className="font-medium text-foreground">Found: </span>{new Date(paper.created_at).toLocaleString()}</p>
                    )}
                  </div>
                </details>
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
