import { useState } from "react";
import type { CalibrationBin } from "../types";
import { pct } from "../format";
import { scale, ticks, useWidth } from "./useWidth";

const H = 240;
const M = { top: 16, right: 16, bottom: 40, left: 44 };

/** Reliability diagram: if confidence were honest, every dot would sit on the diagonal. */
export function ReliabilityChart({ bins, title }: { bins: CalibrationBin[]; title: string }) {
  const [ref, width] = useWidth<HTMLDivElement>(400);
  const [hover, setHover] = useState<number | null>(null);
  const x0 = 0.2;
  const x = scale(x0, 1, M.left, width - M.right);
  const y = scale(0, 1, H - M.bottom, M.top);
  const pts = bins.filter((b) => b.n >= 1);
  const line = pts.map((b, i) => `${i ? "L" : "M"}${x(b.mean_confidence)},${y(b.accuracy)}`).join("");
  const h = hover === null ? null : pts[hover];

  return (
    <div className="chart-wrap" ref={ref}>
      <svg className="chart" width={width} height={H} role="img" aria-label={`${title}: accuracy against Laya's stated confidence`}>
        {ticks(0, 1, 0.25).map((t) => (
          <g key={t}>
            <line className={t === 0 ? "baseline" : "gridline"} x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} />
            <text className="tick-label" x={M.left - 8} y={y(t)} dy="0.32em" textAnchor="end">{pct(t)}</text>
          </g>
        ))}
        {[0.2, 0.4, 0.6, 0.8, 1].map((t) => (
          <text key={t} className="tick-label" x={x(t)} y={H - M.bottom + 18} textAnchor="middle">{pct(t)}</text>
        ))}
        <text className="axis-label" x={(M.left + width - M.right) / 2} y={H - 4} textAnchor="middle">Laya's confidence</text>
        <line x1={x(x0)} y1={y(x0)} x2={x(1)} y2={y(1)} stroke="var(--baseline)" strokeDasharray="4 4" />
        <text className="tick-label" x={x(0.62)} y={y(0.62) - 8} transform={`rotate(${-Math.atan2(y(x0) - y(1), x(1) - x(x0)) * 180 / Math.PI} ${x(0.62)} ${y(0.62) - 8})`}
              textAnchor="middle">honest confidence</text>
        <path d={line} fill="none" stroke="var(--classifier)" strokeWidth={2} />
        {pts.map((b, i) => (
          <g key={i} onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)}>
            <circle cx={x(b.mean_confidence)} cy={y(b.accuracy)} r={14} fill="transparent" />
            <circle cx={x(b.mean_confidence)} cy={y(b.accuracy)} r={hover === i ? 6 : 4.5} fill="var(--classifier)" stroke="var(--surface)" strokeWidth={2} />
          </g>
        ))}
      </svg>
      {h && (
        <div className="tooltip" style={{ left: Math.min(x(h.mean_confidence) + 12, width - 190), top: Math.max(0, y(h.accuracy) - 60) }}>
          <b>Confidence {pct(h.bin_low)}–{pct(h.bin_high)}</b><br />
          {h.n} tickets, average confidence {pct(h.mean_confidence)}<br />
          Actually right: <b>{pct(h.accuracy)}</b>
        </div>
      )}
    </div>
  );
}
