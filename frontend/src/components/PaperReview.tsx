import React, { useState, useEffect, useMemo } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

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

function ConfidenceBadge({ tag }: { tag: Tag }) {
  if (tag.verified === true) return <Badge variant="success">Verified</Badge>;
  if (tag.verified === false) return <Badge variant="warning">Unverified</Badge>;
  return <Badge variant="outline">N/A</Badge>;
}

type ReviewHandler = (tagId: number, status: 'accepted' | 'rejected') => void;
type EditHandler = (tagId: number, value: string) => void;

function TagRow({ tag, onReview, onEdit }: { tag: Tag; onReview: ReviewHandler; onEdit: EditHandler }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(tag.edited_value ?? tag.value ?? '');
  const displayValue = tag.edited_value ?? tag.value ?? '(no value)';

  return (
    <details className="rounded-md border border-input bg-card px-3 py-2">
      <summary className="flex cursor-pointer items-center justify-between gap-2 text-sm">
        <span className="font-medium">{displayValue}</span>
        <span className="flex items-center gap-2">
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
          </div>
        )}
      </div>
    </details>
  );
}

function TagGroup({ tagType, tags, onReview, onEdit }: { tagType: string; tags: Tag[]; onReview: ReviewHandler; onEdit: EditHandler }) {
  return (
    <details className="rounded-md border border-input bg-card">
      <summary className="flex cursor-pointer items-center justify-between gap-2 px-4 py-3 font-serif text-base font-semibold">
        <span>{humanizeTagType(tagType)}</span>
        <Badge variant="secondary">{tags.length}</Badge>
      </summary>
      <div className="flex flex-col gap-2 border-t border-input p-3">
        {tags.map((tag) => <TagRow key={tag.id} tag={tag} onReview={onReview} onEdit={onEdit} />)}
      </div>
    </details>
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
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchResults = async () => {
      if (!id) return;
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:8000/papers/${id}/results`);
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

  const handleReview: ReviewHandler = async (tagId, status) => {
    try {
      const response = await fetch(`http://localhost:8000/tags/${tagId}/review`, {
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

  const handleEdit: EditHandler = async (tagId, value) => {
    try {
      const response = await fetch(`http://localhost:8000/tags/${tagId}/edit`, {
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
      const response = await fetch(`http://localhost:8000/papers/${id}/tags`, {
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

      <div className="flex flex-col gap-4">
        {groupByType(visibleTags).map(([tagType, groupTags]) => (
          <TagGroup key={tagType} tagType={tagType} tags={groupTags} onReview={handleReview} onEdit={handleEdit} />
        ))}
      </div>
    </div>
  );
};

export default PaperReview;
