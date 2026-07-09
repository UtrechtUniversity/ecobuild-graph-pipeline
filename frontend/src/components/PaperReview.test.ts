// Assert-based self-check for computeHighlightMatches.
// Run directly: `bun run src/components/PaperReview.test.ts`
import { computeHighlightMatches } from './PaperReview';

function assert(condition: boolean, message: string): void {
  if (!condition) throw new Error(`Assertion failed: ${message}`);
}

const text = 'The building is located in Lisbon. As mandated by the city council in 2019, an extensive green roof was retrofitted onto the structure. As mandated by the city council in 2019, permits were filed.';

// Two tags with distinct, non-overlapping contexts both get highlighted.
const distinct = computeHighlightMatches(text, [
  { id: 1, context: 'located in Lisbon' },
  { id: 2, context: 'an extensive green roof was retrofitted' },
]);
assert(distinct.length === 2, 'expected both distinct contexts to match');
assert(distinct[0].tagId === 1 && distinct[1].tagId === 2, 'expected matches sorted by position');

// Two tags sharing the exact same context string highlight the same (first) occurrence,
// not two different ones — they're the same passage supporting two findings.
const shared = computeHighlightMatches(text, [
  { id: 3, context: 'As mandated by the city council in 2019' },
  { id: 4, context: 'As mandated by the city council in 2019' },
]);
assert(shared.length === 1, 'expected only the first occurrence to be accepted, not both');
assert(shared[0].start === text.indexOf('As mandated by the city council in 2019'), 'expected the first occurrence');

// A tag whose context isn't found in the text at all is silently skipped, not an error.
const missing = computeHighlightMatches(text, [{ id: 5, context: 'nonexistent phrase' }]);
assert(missing.length === 0, 'expected an unmatched context to be dropped, not throw');

// A null context is skipped, not a crash.
const nullContext = computeHighlightMatches(text, [{ id: 6, context: null }]);
assert(nullContext.length === 0, 'expected a null context to be skipped');

console.log('PaperReview highlight-matching self-check passed');
