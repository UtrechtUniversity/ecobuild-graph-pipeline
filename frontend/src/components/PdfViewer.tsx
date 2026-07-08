import React, { useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import type { PDFDocumentProxy } from 'pdfjs-dist';
import { Button } from './ui/button';
import { bboxToOverlayStyle } from './pdfOverlay';
// @ts-expect-error -- Bun-specific import attribute: resolves to the bundled file's served URL.
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs' with { type: 'file' };

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const RENDER_SCALE = 1.5;

export interface PdfTag {
  id: number;
  page_number: number;
  bbox: { x0: number; y0: number; x1: number; y1: number };
}

export function pdfHighlightId(tagId: number): string {
  return `pdf-highlight-${tagId}`;
}

interface PdfViewerProps {
  paperId: string;
  tags: PdfTag[];
  focusTagId: number | null;
  onHighlightClick: (tagId: number) => void;
}

const PdfViewer: React.FC<PdfViewerProps> = ({ paperId, tags, focusTagId, onHighlightClick }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [doc, setDoc] = useState<PDFDocumentProxy | null>(null);
  const [pageNumber, setPageNumber] = useState(0); // 0-indexed, matches pymupdf's page_number
  const [scale, setScale] = useState(RENDER_SCALE);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const loaded = await pdfjsLib.getDocument(`http://localhost:8000/papers/${paperId}/pdf`).promise;
        if (!cancelled) setDoc(loaded);
      } catch (err) {
        console.error(`Failed to load PDF for paper ${paperId}:`, err);
        if (!cancelled) setError('Failed to load PDF.');
      }
    })();
    return () => { cancelled = true; };
  }, [paperId]);

  useEffect(() => {
    if (focusTagId == null) return;
    const tag = tags.find((t) => t.id === focusTagId);
    if (tag) setPageNumber(tag.page_number);
  }, [focusTagId, tags]);

  useEffect(() => {
    if (!doc || !canvasRef.current) return;
    let cancelled = false;
    (async () => {
      const page = await doc.getPage(pageNumber + 1); // pdf.js pages are 1-indexed
      const viewport = page.getViewport({ scale: RENDER_SCALE });
      const canvas = canvasRef.current;
      if (!canvas) return;
      const context = canvas.getContext('2d');
      if (!context) return;
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      await page.render({ canvasContext: context, viewport, canvas }).promise;
      if (cancelled) return;
      setScale(viewport.scale);
      if (focusTagId != null) {
        requestAnimationFrame(() => {
          const el = document.getElementById(pdfHighlightId(focusTagId));
          el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
          if (el) {
            el.classList.add('ring-2', 'ring-primary');
            setTimeout(() => el.classList.remove('ring-2', 'ring-primary'), 1500);
          }
        });
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc, pageNumber]);

  if (error) return <p className="text-sm text-destructive">{error}</p>;
  if (!doc) return <p className="text-sm text-muted-foreground">Loading PDF…</p>;

  const pageTags = tags.filter((t) => t.page_number === pageNumber);

  return (
    <div className="flex flex-col gap-2 rounded-md border border-input bg-card p-3">
      <div className="flex items-center justify-between text-sm">
        <span>Page {pageNumber + 1} of {doc.numPages}</span>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" disabled={pageNumber === 0} onClick={() => setPageNumber((p) => p - 1)}>Prev</Button>
          <Button size="sm" variant="outline" disabled={pageNumber >= doc.numPages - 1} onClick={() => setPageNumber((p) => p + 1)}>Next</Button>
        </div>
      </div>
      <div className="relative w-fit overflow-auto">
        <canvas ref={canvasRef} />
        {pageTags.map((tag) => {
          const style = bboxToOverlayStyle(tag.bbox, scale);
          return (
            <div
              key={tag.id}
              id={pdfHighlightId(tag.id)}
              onClick={() => onHighlightClick(tag.id)}
              className="absolute cursor-pointer border-2 border-warning bg-warning/20 hover:bg-warning/40"
              style={style}
            />
          );
        })}
      </div>
    </div>
  );
};

export default PdfViewer;
