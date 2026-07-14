import React, { useEffect, useMemo, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import PlotHistogram from './PlotHistogram';
import SearchableSelect from './SearchableSelect';

interface StatBar {
  name: string;
  paper_count: number;
}

interface YearBar {
  year: number;
  paper_count: number;
}

function useStat<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`http://localhost:8000${path}`)
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then(setData)
      .catch(() => setError('Failed to reach the backend.'));
  }, [path]);

  return { data, error };
}

const StatCard: React.FC<{ title: string; path: string }> = ({ title, path }) => {
  const { data, error } = useStat<StatBar[]>(path);
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {error && <p className="text-sm text-destructive">{error}</p>}
        {!error && (data === null
          ? <p className="text-sm text-muted-foreground">Loading…</p>
          : <PlotHistogram data={data} defaultTitle={title} countAxisLabel="Paper count" />)}
      </CardContent>
    </Card>
  );
};

// "ds:<name>" / "es:<name>" — cheap enough for the handful of DS/ES names here,
// avoids a generic option type just to carry the field alongside the value.
const PapersByYearCard: React.FC = () => {
  const { data: designStrategies } = useStat<StatBar[]>('/stats/design-strategies');
  const { data: ecosystemServices } = useStat<StatBar[]>('/stats/ecosystem-services');
  const options = useMemo(() => [
    ...(designStrategies ?? []).map((d) => ({ value: `ds:${d.name}`, label: `DS: ${d.name}` })),
    ...(ecosystemServices ?? []).map((d) => ({ value: `es:${d.name}`, label: `ES: ${d.name}` })),
  ], [designStrategies, ecosystemServices]);

  const [selected, setSelected] = useState('');
  const [data, setData] = useState<YearBar[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selected) { setData(null); return; }
    const field = selected.startsWith('ds:') ? 'design_strategy' : 'ecosystem_service';
    const name = selected.slice(3);
    setData(null);
    setError(null);
    fetch(`http://localhost:8000/stats/papers-by-year?field=${field}&value=${encodeURIComponent(name)}`)
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then(setData)
      .catch(() => setError('Failed to reach the backend.'));
  }, [selected]);

  const bars = useMemo(() => (data ?? []).map((d) => ({ name: String(d.year), paper_count: d.paper_count })), [data]);
  const selectedLabel = options.find((o) => o.value === selected)?.label ?? '';

  return (
    <Card>
      <CardHeader>
        <CardTitle>Papers per year</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <SearchableSelect
          options={options}
          value={selected}
          onChange={setSelected}
          placeholder="Search a design strategy or ecosystem service…"
        />
        {error && <p className="text-sm text-destructive">{error}</p>}
        {!error && !selected && <p className="text-sm text-muted-foreground">Pick a design strategy or ecosystem service.</p>}
        {!error && selected && (data === null
          ? <p className="text-sm text-muted-foreground">Loading…</p>
          : <PlotHistogram key={selected} data={bars} defaultTitle={selectedLabel} countAxisLabel="Paper count" defaultHorizontal={false} />)}
      </CardContent>
    </Card>
  );
};

const MetaStudy: React.FC = () => (
  <div className="flex flex-col gap-8">
    <StatCard title="Studies per design strategy (DS)" path="/stats/design-strategies" />
    <StatCard title="Studies per ecosystem service (ES)" path="/stats/ecosystem-services" />
    <PapersByYearCard />
  </div>
);

export default MetaStudy;
