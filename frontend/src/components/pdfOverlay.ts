/**
 * pymupdf's bbox is already top-left-origin, in the same point space as a PDF
 * page's un-rotated mediabox — the same space pdf.js's own page dimensions are
 * reported in. So placing an overlay is a plain scale, no y-flip: confirmed by
 * comparing pymupdf's reported page/word rects against pdf.js's viewport for
 * the same file (both agree on page size and where "near the top" sits).
 *
 * Kept dependency-free (no pdfjs-dist import) so it's testable outside a
 * browser — pdfjs-dist's main build references browser globals (DOMMatrix)
 * at import time and can't load in a plain JS runtime.
 */
export function bboxToOverlayStyle(
  bbox: { x0: number; y0: number; x1: number; y1: number },
  scale: number,
): { left: number; top: number; width: number; height: number } {
  return {
    left: bbox.x0 * scale,
    top: bbox.y0 * scale,
    width: (bbox.x1 - bbox.x0) * scale,
    height: (bbox.y1 - bbox.y0) * scale,
  };
}
