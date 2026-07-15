import { API_BASE } from '../config';
import React, { useState, useEffect, useMemo } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import PdfViewer from './PdfViewer';

interface Tag {
  id: number;
  tag_type: string;
  group_id: number | null;
  field: string | null;
  value: string | null;
  category: string | null;
  anchor_text: string | null;
  context: string | null;
  match_score: number | null;
  rationale: string | null;
  verified: boolean | null;
  extra_data: Record<string, unknown> | null;
  page_number: number | null;
  bbox: { x0: number; y0: number; x1: number; y1: number } | null;
  review_status: 'pending' | 'accepted' | 'rejected' | 'edited';
  edited_value: string | null;
  added_manually: boolean;
}

function humanizeTagType(tagType: string): string {
  return tagType.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

function groupByType(tags: Tag[]): [string, Tag[]][] {
  const groups = new Map<string, Tag[]>();
  for (const tag of tags) {
    const group = groups.get(tag.tag_type) ?? [];
    group.push(tag);
    groups.set(tag.tag_type, group);
  }
  return Array.from(groups.entries());
}

// Groups by displayed value so recurring labels (same value, several
// occurrences in the paper) collapse into one card with a count.
function groupByValue(tags: Tag[]): [string, Tag[]][] {
  const groups = new Map<string, Tag[]>();
  for (const tag of tags) {
    const key = tag.edited_value ?? tag.value ?? '(no value)';
    const group = groups.get(key) ?? [];
    group.push(tag);
    groups.set(key, group);
  }
  return Array.from(groups.entries());
}

function ConfidenceBadge({ tag }: { tag: Tag }) {
  if (tag.verified === true) return <Badge variant="success">Verified</Badge>;
  if (tag.verified === false) return <Badge variant="warning">Unverified</Badge>;
  return <Badge variant="outline">N/A</Badge>;
}

type ReviewHandler = (tagId: number, status: 'accepted' | 'rejected') => void;
type EditHandler = (tagId: number, value: string) => void;

function tagRowId(tagId: number): string {
  return `tag-row-${tagId}`;
}

function tagGroupId(tagType: string): string {
  return `tag-group-${tagType}`;
}

function labelGroupId(value: string): string {
  return `label-group-${value}`;
}

function textHighlightId(tagId: number): string {
  return `text-highlight-${tagId}`;
}

const FLASH_CLASSES = ['ring-2', 'ring-primary', 'bg-primary/20'];

// Only one element flashes at a time — rapid hovers/clicks (e.g. sweeping
// across cards) would otherwise leave several highlights glowing at once,
// each waiting on its own now-orphaned timeout.
let activeFlash: { el: Element; timeoutId: ReturnType<typeof setTimeout> } | null = null;

function flash(el: Element | null) {
  if (activeFlash) {
    activeFlash.el.classList.remove(...FLASH_CLASSES);
    clearTimeout(activeFlash.timeoutId);
    activeFlash = null;
  }
  if (!el) return;
  el.classList.add(...FLASH_CLASSES);
  const timeoutId = setTimeout(() => {
    el.classList.remove(...FLASH_CLASSES);
    activeFlash = null;
  }, 1500);
  activeFlash = { el, timeoutId };
}

function TagRow({ tag, onReview, onEdit, onLocate, onLocateInPdf, onHover }: {
  tag: Tag; onReview: ReviewHandler; onEdit: EditHandler; onLocate: (tagId: number) => void; onLocateInPdf: (tagId: number) => void;
  onHover: (tagId: number) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(tag.edited_value ?? tag.value ?? '');
  const displayValue = tag.edited_value ?? tag.value ?? '(no value)';

  return (
    <details id={tagRowId(tag.id)} className="rounded-md border border-input bg-card px-3 py-2" onMouseEnter={() => onHover(tag.id)}>
      <summary className="flex cursor-pointer items-center justify-between gap-2 text-sm">
        <span className="flex min-w-0 items-center gap-2 font-medium">
          <span className="truncate">{displayValue}</span>
          {tag.match_score !== null && <Badge variant="outline" className="shrink-0">{tag.match_score.toFixed(2)}</Badge>}
        </span>
        <span className="flex shrink-0 flex-wrap items-center justify-end gap-1">
          {tag.added_manually && <Badge variant="secondary">Added</Badge>}
          {tag.review_status === 'edited' && <Badge variant="secondary">Edited</Badge>}
          {tag.review_status === 'accepted' && <Badge variant="success">Accepted</Badge>}
          {tag.category && <Badge variant="outline">{tag.category}</Badge>}
          {tag.field && <Badge variant="outline">{tag.field}</Badge>}
          <ConfidenceBadge tag={tag} />
        </span>
      </summary>
      <div className="mt-2 flex flex-col gap-2 border-t border-input pt-2 text-xs text-muted-foreground">
        {tag.edited_value && <p><span className="font-medium text-foreground">Originally: </span>{tag.value}</p>}
        {tag.context && <p><span className="font-medium text-foreground">Context: </span>{tag.context}</p>}
        {tag.anchor_text && <p><span className="font-medium text-foreground">Anchor: </span>{tag.anchor_text}</p>}
        {tag.rationale && <p><span className="font-medium text-foreground">Rationale: </span>{tag.rationale}</p>}
        {tag.match_score !== null && (
          <p><span className="font-medium text-foreground">Match score: </span>{tag.match_score.toFixed(2)}</p>
        )}
        {editing ? (
          <div className="flex gap-2 pt-1">
            <input
              className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-xs"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              autoFocus
            />
            <Button size="sm" variant="outline" onClick={() => { onEdit(tag.id, draft); setEditing(false); }}>Save</Button>
            <Button size="sm" variant="outline" onClick={() => setEditing(false)}>Cancel</Button>
          </div>
        ) : (
          <div className="flex gap-2 pt-1">
            <Button size="sm" variant="outline" onClick={() => onReview(tag.id, 'accepted')}>Accept</Button>
            <Button size="sm" variant="outline" onClick={() => onReview(tag.id, 'rejected')}>Reject</Button>
            <Button size="sm" variant="outline" onClick={() => setEditing(true)}>Edit</Button>
            {tag.context && (
              <Button size="sm" variant="outline" onClick={() => onLocate(tag.id)}>Locate in text</Button>
            )}
            {tag.page_number !== null && tag.bbox !== null && (
              <Button size="sm" variant="outline" onClick={() => onLocateInPdf(tag.id)}>Locate in PDF</Button>
            )}
          </div>
        )}
      </div>
    </details>
  );
}

function TagGroup({ id, title, tags, onReview, onEdit, onLocate, onLocateInPdf, onHover }: {
  id: string; title: string; tags: Tag[]; onReview: ReviewHandler; onEdit: EditHandler; onLocate: (tagId: number) => void; onLocateInPdf: (tagId: number) => void;
  onHover: (tagId: number) => void;
}) {
  return (
    <details id={id} className="rounded-md border border-input bg-card">
      <summary className="flex cursor-pointer items-center justify-between gap-2 px-4 py-3 font-serif text-base font-semibold">
        <span>{title}</span>
        <Badge variant="secondary">{tags.length}</Badge>
      </summary>
      <div className="flex flex-col gap-2 border-t border-input p-3">
        {tags.map((tag) => <TagRow key={tag.id} tag={tag} onReview={onReview} onEdit={onEdit} onLocate={onLocate} onLocateInPdf={onLocateInPdf} onHover={onHover} />)}
      </div>
    </details>
  );
}

// One card per recurring label value. Sorted by match score so the
// best-supported occurrence leads. Accepting any occurrence accepts the
// whole group — a label is either validated for this paper or it isn't,
// occurrence-by-occurrence acceptance would just be busywork.
function ValueGroup({ value, tags, onReview, onAcceptGroup, onEdit, onLocate, onLocateInPdf, onHover }: {
  value: string; tags: Tag[]; onReview: ReviewHandler; onAcceptGroup: (tagIds: number[]) => void;
  onEdit: EditHandler; onLocate: (tagId: number) => void; onLocateInPdf: (tagId: number) => void;
  onHover: (tagId: number) => void;
}) {
  const sorted = useMemo(() => [...tags].sort((a, b) => (b.match_score ?? -1) - (a.match_score ?? -1)), [tags]);
  const groupIds = useMemo(() => tags.map((t) => t.id), [tags]);
  const handleReview: ReviewHandler = (tagId, status) => {
    if (status === 'accepted') onAcceptGroup(groupIds);
    else onReview(tagId, status);
  };

  return (
    <TagGroup
      id={labelGroupId(value)} title={value} tags={sorted}
      onReview={handleReview} onEdit={onEdit} onLocate={onLocate} onLocateInPdf={onLocateInPdf} onHover={onHover}
    />
  );
}

// ponytail: highlights the first occurrence of each tag's context in the text.
// Two different tags can legitimately share identical context (the same
// sentence supporting two findings) — they highlight the same spot, which is
// correct. What this doesn't handle: an unrelated but textually identical
// phrase appearing earlier elsewhere in the paper, which would "steal" the
// highlight. Fixing that needs the extraction pipeline to capture a character
// offset when it resolves each anchor (context_resolver.find_anchor_in_text)
// and carry it through to the tag row — a pipeline change, not a UI one.
export interface HighlightMatch { start: number; end: number; tagId: number }

/** Pure matching logic, exported so it's testable without a JSX/DOM runtime. */
export function computeHighlightMatches(text: string, tags: Pick<Tag, 'id' | 'context'>[]): HighlightMatch[] {
  const matches: HighlightMatch[] = [];
  for (const tag of tags) {
    if (!tag.context) continue;
    const start = text.indexOf(tag.context);
    if (start === -1) continue;
    matches.push({ start, end: start + tag.context.length, tagId: tag.id });
  }
  matches.sort((a, b) => a.start - b.start);

  const accepted: HighlightMatch[] = [];
  let lastEnd = -1;
  for (const m of matches) {
    if (m.start >= lastEnd) {
      accepted.push(m);
      lastEnd = m.end;
    }
  }
  return accepted;
}

function buildHighlightSegments(text: string, tags: Tag[], onHighlightClick: (tagId: number) => void): React.ReactNode[] {
  const accepted = computeHighlightMatches(text, tags);
  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  for (const m of accepted) {
    if (m.start > cursor) nodes.push(text.slice(cursor, m.start));
    nodes.push(
      <mark
        key={m.tagId}
        id={textHighlightId(m.tagId)}
        className="cursor-pointer rounded bg-warning/30 px-0.5 hover:bg-warning/50"
        onClick={() => onHighlightClick(m.tagId)}
      >
        {text.slice(m.start, m.end)}
      </mark>
    );
    cursor = m.end;
  }
  if (cursor < text.length) nodes.push(text.slice(cursor));
  return nodes;
}

function PaperText({ text, tags, onHighlightClick }: { text: string; tags: Tag[]; onHighlightClick: (tagId: number) => void }) {
  const segments = useMemo(() => buildHighlightSegments(text, tags, onHighlightClick), [text, tags, onHighlightClick]);
  return (
    <div className="max-h-150 overflow-auto whitespace-pre-wrap rounded-md border border-input bg-card p-4 text-sm leading-relaxed">
      {segments}
    </div>
  );
}

function AddTagForm({ tagTypes, onAdd }: { tagTypes: string[]; onAdd: (tagType: string, value: string) => void }) {
  const [tagType, setTagType] = useState(tagTypes[0] ?? '');
  const [value, setValue] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!tagType || !value.trim()) return;
    onAdd(tagType, value.trim());
    setValue('');
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2 rounded-md border border-input bg-card p-3">
      <select
        className="rounded-md border border-input bg-background px-2 py-1 text-sm"
        value={tagType}
        onChange={(e) => setTagType(e.target.value)}
      >
        {tagTypes.map((t) => <option key={t} value={t}>{humanizeTagType(t)}</option>)}
      </select>
      <input
        className="flex-1 rounded-md border border-input bg-background px-2 py-1 text-sm"
        placeholder="Value the extractor missed…"
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
      <Button type="submit" size="sm" variant="outline">Add tag</Button>
    </form>
  );
}

const PaperReview: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [tags, setTags] = useState<Tag[] | null>(null);
  const [text, setText] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [focusPdfTagId, setFocusPdfTagId] = useState<number | null>(null);
  const [viewTab, setViewTab] = useState<'text' | 'pdf'>('text');

  useEffect(() => {
    const fetchResults = async () => {
      if (!id) return;
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE}/papers/${id}/results`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: { tags: Tag[] } = await response.json();
        setTags(data.tags);
        setError(null);
      } catch (err) {
        console.error(`Failed to fetch results for paper ${id}:`, err);
        setError(`Failed to load extraction results for paper ${id}.`);
      } finally {
        setLoading(false);
      }
    };

    fetchResults();
  }, [id]);

  useEffect(() => {
    const fetchText = async () => {
      if (!id) return;
      try {
        const response = await fetch(`${API_BASE}/papers/${id}/text`);
        if (!response.ok) return; // no extracted text yet — text panel just doesn't render
        const data: { text: string } = await response.json();
        setText(data.text);
      } catch (err) {
        console.error(`Failed to fetch extracted text for paper ${id}:`, err);
      }
    };

    fetchText();
  }, [id]);

  const revealTag = (tagId: number) => {
    const tag = tags?.find((t) => t.id === tagId);
    if (!tag) return;
    const groupId = tag.tag_type === 'label' ? labelGroupId(tag.edited_value ?? tag.value ?? '(no value)') : tagGroupId(tag.tag_type);
    const groupEl = document.getElementById(groupId) as HTMLDetailsElement | null;
    const rowEl = document.getElementById(tagRowId(tagId)) as HTMLDetailsElement | null;
    if (groupEl) groupEl.open = true;
    if (rowEl) rowEl.open = true;
    rowEl?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    flash(rowEl);
  };

  // Scrolls/flashes the text highlight if that panel happens to be visible;
  // a no-op otherwise (nothing to see on a `hidden` panel). Doesn't touch
  // which tab is active — used for hover, where flipping tabs would fight
  // the user for control.
  const peekHighlight = (tagId: number) => {
    const el = document.getElementById(textHighlightId(tagId));
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    flash(el);
  };

  const peekInPdf = (tagId: number) => setFocusPdfTagId(tagId);

  const revealHighlight = (tagId: number) => {
    setViewTab('text');
    // The text panel may have been hidden (inactive tab) a moment ago; defer
    // to the next frame so it's laid out before we scroll to it.
    requestAnimationFrame(() => peekHighlight(tagId));
  };

  const revealInPdf = (tagId: number) => {
    setViewTab('pdf');
    peekInPdf(tagId);
  };

  // Hovering a card previews its location in both panels without stealing
  // the active tab, so browsing rows doesn't yank the viewer around.
  const handleHoverTag = (tagId: number) => {
    peekHighlight(tagId);
    peekInPdf(tagId);
  };

  const handleReview: ReviewHandler = async (tagId, status) => {
    try {
      const response = await fetch(`${API_BASE}/tags/${tagId}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      setTags((prev) => prev && prev.map((t) => (t.id === tagId ? { ...t, review_status: status } : t)));
    } catch (err) {
      console.error(`Failed to update review status for tag ${tagId}:`, err);
    }
  };

  // Accepting one occurrence of a recurring label accepts every occurrence
  // of that label — there's no per-tag review endpoint for "accept a set",
  // so this just fires the single-tag endpoint for each id in parallel.
  const handleAcceptGroup = async (tagIds: number[]) => {
    try {
      await Promise.all(tagIds.map((tagId) =>
        fetch(`${API_BASE}/tags/${tagId}/review`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'accepted' }),
        }),
      ));
      setTags((prev) => prev && prev.map((t) => (tagIds.includes(t.id) ? { ...t, review_status: 'accepted' } : t)));
    } catch (err) {
      console.error('Failed to accept label group:', err);
    }
  };

  const handleEdit: EditHandler = async (tagId, value) => {
    try {
      const response = await fetch(`${API_BASE}/tags/${tagId}/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      setTags((prev) => prev && prev.map((t) => (t.id === tagId ? { ...t, edited_value: value, review_status: 'edited' } : t)));
    } catch (err) {
      console.error(`Failed to edit tag ${tagId}:`, err);
    }
  };

  const handleAdd = async (tagType: string, value: string) => {
    if (!id) return;
    try {
      const response = await fetch(`${API_BASE}/papers/${id}/tags`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tag_type: tagType, value }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data: { tags: Tag[] } = await response.json();
      setTags(data.tags);
    } catch (err) {
      console.error(`Failed to add a tag for paper ${id}:`, err);
    }
  };

  // Rejected tags are hidden from this default view but never deleted server-side.
  const visibleTags = (tags ?? []).filter((tag) => tag.review_status !== 'rejected');
  const knownTagTypes = useMemo(() => Array.from(new Set((tags ?? []).map((t) => t.tag_type))), [tags]);
  const pdfTags = useMemo(
    () => visibleTags
      .filter((t): t is Tag & { page_number: number; bbox: NonNullable<Tag['bbox']> } => t.page_number !== null && t.bbox !== null)
      .map((t) => ({ id: t.id, page_number: t.page_number, bbox: t.bbox })),
    [visibleTags],
  );
  const labelValueGroups = useMemo(
    () => groupByValue(visibleTags.filter((t) => t.tag_type === 'label')),
    [visibleTags],
  );
  const otherTypeGroups = useMemo(
    () => groupByType(visibleTags.filter((t) => t.tag_type !== 'label')),
    [visibleTags],
  );

  return (
    <div className="flex flex-col gap-6">
      <Link to="/" className="inline-flex w-fit items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="size-4" /> Back to papers
      </Link>

      <h2 className="font-serif text-2xl font-semibold">Paper #{id} — extraction results</h2>

      {loading && <p className="text-sm text-muted-foreground">Loading results…</p>}
      {error && <p className="text-sm text-destructive">{error}</p>}
      {tags && tags.length === 0 && <p className="text-sm text-muted-foreground">No tags extracted for this paper.</p>}

      {knownTagTypes.length > 0 && <AddTagForm tagTypes={knownTagTypes} onAdd={handleAdd} />}

      <div className="flex gap-4">
        {labelValueGroups.length > 0 && (
          <div className="flex w-80 shrink-0 flex-col gap-3">
            <h3 className="font-serif text-lg font-semibold">Labels</h3>
            {labelValueGroups.map(([value, groupTags]) => (
              <ValueGroup
                key={value} value={value} tags={groupTags}
                onReview={handleReview} onAcceptGroup={handleAcceptGroup} onEdit={handleEdit}
                onLocate={revealHighlight} onLocateInPdf={revealInPdf} onHover={handleHoverTag}
              />
            ))}
          </div>
        )}

        <div className="flex flex-1 flex-col gap-2">
          <div className="flex gap-2">
            <Button size="sm" variant={viewTab === 'text' ? 'default' : 'outline'} onClick={() => setViewTab('text')}>Text view</Button>
            <Button size="sm" variant={viewTab === 'pdf' ? 'default' : 'outline'} onClick={() => setViewTab('pdf')}>PDF view</Button>
          </div>
          <div className={viewTab === 'text' ? undefined : 'hidden'}>
            {text && <PaperText text={text} tags={visibleTags} onHighlightClick={revealTag} />}
          </div>
          <div className={viewTab === 'pdf' ? undefined : 'hidden'}>
            {id && <PdfViewer paperId={id} tags={pdfTags} focusTagId={focusPdfTagId} onHighlightClick={revealTag} />}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        {otherTypeGroups.map(([tagType, groupTags]) => (
          <TagGroup
            key={tagType} id={tagGroupId(tagType)} title={humanizeTagType(tagType)} tags={groupTags}
            onReview={handleReview} onEdit={handleEdit} onLocate={revealHighlight} onLocateInPdf={revealInPdf} onHover={handleHoverTag}
          />
        ))}
      </div>
    </div>
  );
};

export default PaperReview;
