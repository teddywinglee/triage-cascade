import { useState } from "react";
import type { LanguageRow } from "../types";
import { LANG_NAMES, pct } from "../format";
import { scale, ticks, useWidth } from "./useWidth";

const ROW = 30;
const M = { top: 12, right: 24, bottom: 32, left: 120 };

/** One row per language: Laya's accuracy and Gemma's, joined so the gap is the story. */
export function DumbbellChart({ rows, field }: { rows: LanguageRow[]; field: "sentiment" | "department" }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const data = [...rows].sort((a, b) => (a[`classifier_${field}`] as number) - (b[`classifier_${field}`] as number));
  const H = M.top + data.length * ROW + M.bottom;
  const x = scale(0.4, 1, M.left, width - M.right);
  const h = hover === null ? null : data[hover];

  return (
    <div className="chart-wrap" ref={ref}>
      <svg className="chart" width={width} height={H} role="img" aria-label={`${field} accuracy by language, Laya and Gemma`}>
        {ticks(0.4, 1, 0.1).map((t) => (
          <g key={t}>
            <line className={t === 0.4 ? "baseline" : "gridline"} x1={x(t)} x2={x(t)} y1={M.top} y2={H - M.bottom} />
            <text className="tick-label" x={x(t)} y={H - M.bottom + 18} textAnchor="middle">{pct(t)}</text>
          </g>
        ))}
        {data.map((r, i) => {
          const cy = M.top + i * ROW + ROW / 2;
          const l = r[`classifier_${field}`] as number;
          const g = r[`llm_${field}`];
          return (
            <g key={r.lang} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
              <rect x={0} y={cy - ROW / 2} width={width} height={ROW} fill={hover === i ? "var(--wash)" : "transparent"} />
              <text x={M.left - 12} y={cy} dy="0.32em" textAnchor="end" style={{ fill: "var(--ink-2)", fontSize: 13 }}>
                {LANG_NAMES[r.lang] ?? r.lang}
              </text>
              {g !== null && <line x1={x(Math.max(l, 0.4))} x2={x(Math.max(g, 0.4))} y1={cy} y2={cy} stroke="var(--baseline)" strokeWidth={2} />}
              <circle cx={x(Math.max(l, 0.4))} cy={cy} r={6} fill="var(--classifier)" stroke="var(--surface)" strokeWidth={2} />
              {g !== null && <circle cx={x(Math.max(g, 0.4))} cy={cy} r={6} fill="var(--llm)" stroke="var(--surface)" strokeWidth={2} />}
            </g>
          );
        })}
      </svg>
      {h && (
        <div className="tooltip" style={{ left: Math.min(x(h[`classifier_${field}`] as number) + 16, width - 200), top: M.top + (hover ?? 0) * ROW - 52 }}>
          <b>{LANG_NAMES[h.lang] ?? h.lang}</b> ({h.n} tickets)<br />
          Laya: <b>{pct(h[`classifier_${field}`] as number)}</b><br />
          Gemma: <b>{pct(h[`llm_${field}`])}</b>
        </div>
      )}
    </div>
  );
}
