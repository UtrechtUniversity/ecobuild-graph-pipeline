import React, { useEffect, useMemo, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';
import { Button } from './ui/button';

interface HistogramBar {
  name: string;
  paper_count: number;
}

const FONT_OPTIONS = [
  'Arial, sans-serif',
  'Helvetica, sans-serif',
  'Times New Roman, serif',
  'Georgia, serif',
  'Calibri, sans-serif',
];

const MAX_LABEL_LENGTH = 40;

function cropLabel(name: string): string {
  return name.length > MAX_LABEL_LENGTH ? `${name.slice(0, MAX_LABEL_LENGTH - 1)}…` : name;
}

const PlotHistogram: React.FC<{ data: HistogramBar[]; defaultTitle: string; countAxisLabel: string; defaultHorizontal?: boolean }> = ({
  data, defaultTitle, countAxisLabel, defaultHorizontal = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [title, setTitle] = useState(defaultTitle);
  const [barColor, setBarColor] = useState('#2563eb');
  const [fontFamily, setFontFamily] = useState(FONT_OPTIONS[0]);
  const [fontSize, setFontSize] = useState(14);
  const [horizontal, setHorizontal] = useState(defaultHorizontal);
  const [topN, setTopN] = useState(20);
  const [transparentBg, setTransparentBg] = useState(true);
  const [showGrid, setShowGrid] = useState(true);

  const shown = useMemo(() => data.slice(0, topN), [data, topN]);
  // Fixed, deterministic height (more bars = taller in horizontal mode, so
  // labels don't get squashed) — set on both the container div and the
  // Plotly layout so the Card wrapping it never has to guess the chart's
  // size after an async resize.
  const chartHeight = horizontal ? Math.max(320, shown.length * 28 + 120) : 420;

  useEffect(() => {
    if (!containerRef.current || shown.length === 0) return;
    // Plotly renders categorical axes top-to-bottom in array order; reverse so
    // the highest count ends up on top, matching how the data is pre-sorted.
    const ordered = horizontal ? [...shown].reverse() : shown;
    const fullNames = ordered.map((d) => d.name);
    const names = fullNames.map(cropLabel);
    const counts = ordered.map((d) => d.paper_count);
    // hovertext keeps the full name reachable even though the axis label is cropped.
    const trace = horizontal
      ? { x: counts, y: names, type: 'bar', orientation: 'h', marker: { color: barColor }, hovertext: fullNames, hoverinfo: 'text+x' }
      : { x: names, y: counts, type: 'bar', marker: { color: barColor }, hovertext: fullNames, hoverinfo: 'text+y' };

    const layout: Record<string, unknown> = {
      title: { text: title, font: { family: fontFamily, size: fontSize + 4 } },
      font: { family: fontFamily, size: fontSize },
      height: chartHeight,
      paper_bgcolor: transparentBg ? 'rgba(0,0,0,0)' : '#ffffff',
      plot_bgcolor: transparentBg ? 'rgba(0,0,0,0)' : '#ffffff',
      margin: horizontal ? { l: 260, r: 20, t: 60, b: 60 } : { l: 60, r: 20, t: 60, b: 160 },
      xaxis: horizontal
        ? { title: { text: countAxisLabel }, showgrid: showGrid }
        : { tickangle: -45, automargin: true, showgrid: showGrid },
      yaxis: horizontal
        ? { automargin: true, showgrid: false }
        : { title: { text: countAxisLabel }, showgrid: showGrid },
    };

    Plotly.react(containerRef.current, [trace], layout, { displaylogo: false, responsive: true });
  }, [shown, title, barColor, fontFamily, fontSize, horizontal, transparentBg, showGrid, countAxisLabel, chartHeight]);

  const download = (format: 'png' | 'svg') => {
    if (!containerRef.current) return;
    Plotly.downloadImage(containerRef.current, { format, filename: title.trim() || 'histogram', width: 1200, height: 800, scale: 2 });
  };

  if (data.length === 0) {
    return <p className="text-sm text-muted-foreground">No data yet.</p>;
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-3 text-xs">
        <label className="flex flex-col gap-1">
          Title
          <input value={title} onChange={(e) => setTitle(e.target.value)} className="h-8 rounded-md border border-input bg-card px-2 text-sm" />
        </label>
        <label className="flex flex-col gap-1">
          Bar color
          <input type="color" value={barColor} onChange={(e) => setBarColor(e.target.value)} className="h-8 w-12 rounded-md border border-input bg-card" />
        </label>
        <label className="flex flex-col gap-1">
          Font
          <select value={fontFamily} onChange={(e) => setFontFamily(e.target.value)} className="h-8 rounded-md border border-input bg-card px-2 text-sm">
            {FONT_OPTIONS.map((f) => <option key={f} value={f}>{f.split(',')[0]}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          Font size
          <input type="number" min={8} max={32} value={fontSize} onChange={(e) => setFontSize(Number(e.target.value))} className="h-8 w-16 rounded-md border border-input bg-card px-2 text-sm" />
        </label>
        <label className="flex flex-col gap-1">
          Show top N
          <input type="number" min={1} max={data.length} value={topN} onChange={(e) => setTopN(Number(e.target.value))} className="h-8 w-16 rounded-md border border-input bg-card px-2 text-sm" />
        </label>
        <label className="flex items-center gap-1.5 pb-1.5">
          <input type="checkbox" checked={horizontal} onChange={(e) => setHorizontal(e.target.checked)} />
          Horizontal bars
        </label>
        <label className="flex items-center gap-1.5 pb-1.5">
          <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
          Gridlines
        </label>
        <label className="flex items-center gap-1.5 pb-1.5">
          <input type="checkbox" checked={transparentBg} onChange={(e) => setTransparentBg(e.target.checked)} />
          Transparent background
        </label>
        <div className="ml-auto flex gap-2 pb-0.5">
          <Button size="sm" variant="outline" onClick={() => download('png')}>Download PNG</Button>
          <Button size="sm" variant="outline" onClick={() => download('svg')}>Download SVG</Button>
        </div>
      </div>
      <div ref={containerRef} style={{ height: chartHeight }} />
    </div>
  );
};

export default PlotHistogram;
