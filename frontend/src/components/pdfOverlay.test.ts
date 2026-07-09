// Assert-based self-check for the PDF bbox -> overlay coordinate math.
// Run directly: `bun run src/components/pdfOverlay.test.ts`
//
// The numbers here are real, taken from cross-checking pymupdf's search_for()
// output against pdfjs-dist's page viewport for the same PDF/page (see the
// PR description for #63): pymupdf's bbox is already top-left-origin, in the
// same point space pdf.js reports for an unrotated page, so placing an
// overlay is a plain scale with no y-flip.
import { bboxToOverlayStyle } from './pdfOverlay';

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`Assertion failed: ${message}`);
}

// pymupdf: page.search_for("Introduction") on a real paper's first page.
const bbox = { x0: 49.83266067504883, y0: 462.60675048828125, x1: 97.8365707397461, y1: 475.6378479003906 };
// pdf.js: page.getViewport({ scale: 2.0 }).scale for the same PDF/page (595.276 x 793.701 pt, unrotated).
const scale = 2;

const style = bboxToOverlayStyle(bbox, scale);
assert(style.left === bbox.x0 * 2, 'left should be a plain x0 scale, no offset');
assert(style.top === bbox.y0 * 2, 'top should be a plain y0 scale, no y-flip');
assert(Math.abs(style.width - 96.00782012939453) < 1e-6, 'width should be (x1-x0) scaled');
assert(Math.abs(style.height - 26.06219482421875) < 1e-6, 'height should be (y1-y0) scaled');

// A word positioned in the lower half of the page (like "Introduction" on this
// paper's dense two-column first page) should overlay in the lower half of a
// scaled canvas — sanity check against a page rendered at this scale.
const scaledPageHeight = 793.7009887695312 * scale;
assert(style.top / scaledPageHeight > 0.5, 'expected the overlay to land in the lower half of the page, matching where this heading actually sits');

console.log('pdfOverlay bbox-to-style self-check passed');
