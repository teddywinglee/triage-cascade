import { useState } from "react";
import type { PolicyMetrics } from "../types";
import { pct } from "../format";
import { scale, ticks, useWidth } from "./useWidth";

interface Props {
  curve: PolicyMetrics[];
  current: PolicyMetrics;
  classifierOnly: PolicyMetrics;
  llmOnly: PolicyMetrics;
  metric: (m: PolicyMetrics) => number;
  metricLabel: string;
  minutesFor: (m: PolicyMetrics) => string;
  onPick: (confidence: number) => void;
}

const H = 300;
const M = { top: 20, right: 28, bottom: 40, left: 48 };

/** Escalation rate (x) vs. a quality metric (y). Each curve point is a confidence threshold. */
export function TradeoffChart({ curve, current, classifierOnly, llmOnly, metric, metricLabel, minutesFor, onPick }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const pts = [...curve].sort((a, b) => a.escalation_rate - b.escalation_rate || (a.confidence ?? 0) - (b.confidence ?? 0));
  const ys = [...pts.map(metric), metric(classifierOnly), metric(llmOnly)];
  const yMin = Math.max(0, Math.floor((Math.min(...ys) - 0.05) * 10) / 10);
  const yMax = Math.min(1, Math.ceil((Math.max(...ys) + 0.02) * 10) / 10);
  const x = scale(0, 1, M.left, width - M.right);
  const y = scale(yMin, yMax, H - M.bottom, M.top);
  const yStep = yMax - yMin > 0.5 ? 0.2 : 0.1;

  const line = pts.map((p, i) => `${i ? "L" : "M"}${x(p.escalation_rate).toFixed(1)},${y(metric(p)).toFixed(1)}`).join("");
  const h = hover === null ? null : pts[hover];

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left + M.left;
    let best = 0;
    pts.forEach((p, i) => { if (Math.abs(x(p.escalation_rate) - px) < Math.abs(x(pts[best].escalation_rate) - px)) best = i; });
    setHover(best);
  };

  const cx = x(current.escalation_rate), cy = y(metric(current));
  const curLabelLeft = cx > width - 160;

  return (
    <div className="chart-wrap" ref={ref}>
      <svg className="chart" width={width} height={H} role="img"
           aria-label={`${metricLabel} against share of tickets escalated to Gemma`}>
        {ticks(yMin, yMax, yStep).map((t) => (
          <g key={t}>
            <line className={t === yMin ? "baseline" : "gridline"} x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} />
            <text className="tick-label" x={M.left - 8} y={y(t)} dy="0.32em" textAnchor="end">{pct(t)}</text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} className="tick-label" x={x(t)} y={H - M.bottom + 18} textAnchor="middle">{pct(t)}</text>
        ))}
        <text className="axis-label" x={(M.left + width - M.right) / 2} y={H - 4} textAnchor="middle">Tickets sent to Gemma</text>
        <text className="axis-label" x={M.left} y={10}>{metricLabel}</text>

        {h && <line x1={x(h.escalation_rate)} x2={x(h.escalation_rate)} y1={M.top} y2={H - M.bottom} stroke="var(--baseline)" strokeDasharray="3 3" />}

        <path d={line} fill="none" stroke="var(--routed)" strokeWidth={2} strokeLinejoin="round" />
        {pts.map((p, i) => (
          <circle key={i} cx={x(p.escalation_rate)} cy={y(metric(p))} r={hover === i ? 5 : 3}
                  fill="var(--routed)" stroke="var(--surface)" strokeWidth={2} />
        ))}

        {/* Baselines at both ends */}
        <circle cx={x(classifierOnly.escalation_rate)} cy={y(metric(classifierOnly))} r={6} fill="var(--classifier)" stroke="var(--surface)" strokeWidth={2} />
        <text className="direct-label" x={x(0) + 10} y={y(metric(classifierOnly)) + 16}>Laya only</text>
        <circle cx={x(llmOnly.escalation_rate)} cy={y(metric(llmOnly))} r={6} fill="var(--llm)" stroke="var(--surface)" strokeWidth={2} />
        <text className="direct-label" x={x(1) - 10} y={y(metric(llmOnly)) - 12} textAnchor="end">Gemma only</text>

        {/* Current policy */}
        <circle cx={cx} cy={cy} r={9} fill="none" stroke="var(--ink)" strokeWidth={2} />
        <text className="direct-label" x={curLabelLeft ? cx - 14 : cx + 14} y={cy + 22} textAnchor={curLabelLeft ? "end" : "start"}
              style={{ fill: "var(--ink)" }}>
          Your policy
        </text>

        <rect x={M.left} y={M.top} width={Math.max(0, width - M.left - M.right)} height={H - M.top - M.bottom}
              fill="transparent" style={{ cursor: "pointer" }}
              onPointerMove={onMove} onPointerLeave={() => setHover(null)}
              onClick={() => h && onPick(h.confidence ?? 0)} />
      </svg>
      {h && (
        <div className="tooltip" style={{
          left: Math.min(x(h.escalation_rate) + 12, width - 210),
          top: Math.max(0, y(metric(h)) - 70),
        }}>
          <b>Confidence threshold {h.confidence?.toFixed(2)}</b><br />
          {pct(h.escalation_rate)} sent to Gemma<br />
          {metricLabel}: <b>{pct(metric(h), 1)}</b><br />
          Time for {h.n} tickets: {minutesFor(h)}<br />
          <span style={{ color: "var(--muted)" }}>Click to use this threshold</span>
        </div>
      )}
    </div>
  );
}
