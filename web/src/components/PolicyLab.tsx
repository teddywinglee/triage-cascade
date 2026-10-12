import { useState } from "react";
import type { PolicyMetrics, PolicyParams, PolicyResponse } from "../types";
import { minutes, pct } from "../format";
import { TradeoffChart } from "../charts/TradeoffChart";

type MetricKey = "department" | "sentiment" | "churn" | "refund";
const METRICS: Record<MetricKey, { label: string; get: (m: PolicyMetrics) => number }> = {
  department: { label: "Department accuracy", get: (m) => m.department },
  sentiment: { label: "Sentiment accuracy", get: (m) => m.sentiment },
  churn: { label: "Churn threats caught", get: (m) => m.churn.recall },
  refund: { label: "Refund flags that are right", get: (m) => m.refund.precision },
};

interface Props {
  policy: PolicyParams;
  setPolicy: (p: PolicyParams) => void;
  data: PolicyResponse;
  classifierSeconds: number;
  llmSeconds: number;
  setLlmSeconds: (s: number) => void;
  measuredLlmSeconds: number;
}

export function PolicyLab({ policy, setPolicy, data, classifierSeconds, llmSeconds, setLlmSeconds, measuredLlmSeconds }: Props) {
  const [metric, setMetric] = useState<MetricKey>("department");
  const m = METRICS[metric];

  // Time is recomputed here so the "cost of one Gemma call" setting applies everywhere.
  const totalMinutes = (x: PolicyMetrics) => {
    const usesClassifier = x.policy.confidence <= 1;
    return ((usesClassifier ? x.n * classifierSeconds : 0) + x.escalated * llmSeconds) / 60;
  };
  const rows: [string, string, PolicyMetrics][] = [
    ["Laya only", "var(--classifier)", data.classifier_only],
    ["Your policy", "var(--routed)", data.current],
    ["Gemma only", "var(--llm)", data.llm_only],
  ];
  const presets = [
    { label: `Measured (${measuredLlmSeconds.toFixed(1)} s)`, v: measuredLlmSeconds },
    { label: "With thinking (10 s)", v: 10 },
    { label: "Paid API (3 s)", v: 3 },
  ];

  return (
    <>
      <div className="lab">
        <div className="controls">
          <div className="control">
            <label htmlFor="conf">Send to Gemma when Laya is less sure than
              <span className="value">{policy.confidence > 1 ? "always" : policy.confidence.toFixed(2)}</span>
            </label>
            <p className="help">Uses the lower of Laya's sentiment and department confidence.</p>
            <input id="conf" type="range" min={0} max={1} step={0.05} value={Math.min(policy.confidence, 1)}
                   onChange={(e) => setPolicy({ ...policy, confidence: +e.target.value })} />
          </div>

          <div className="control">
            <label className="switch">
              <input type="checkbox" checked={policy.churn_screen !== null}
                     onChange={(e) => setPolicy({ ...policy, churn_screen: e.target.checked ? 0.005 : null })} />
              Double-check possible churn
            </label>
            <p className="help">Laya misses most churn threats on its own, but its score is a useful filter (above 0.005, on negative or neutral tickets).</p>
          </div>

          <div className="control">
            <label className="switch">
              <input type="checkbox" checked={policy.refund_check !== null}
                     onChange={(e) => setPolicy({ ...policy, refund_check: e.target.checked ? 0.5 : null })} />
              Double-check refund claims
            </label>
            <p className="help">Laya finds 93% of refund requests but raises false alarms on other billing complaints.</p>
          </div>

          <div className="control">
            <span className="label">Cost of one Gemma call</span>
            <p className="help">Routing matters more when the expensive model is slower or paid.</p>
            <div className="segmented" role="group" aria-label="Gemma cost presets">
              {presets.map((p) => (
                <button key={p.label} aria-pressed={Math.abs(llmSeconds - p.v) < 1e-6} onClick={() => setLlmSeconds(p.v)}>{p.label}</button>
              ))}
            </div>
            <div className="cost-input">
              <input type="number" min={0.1} step={0.1} value={+llmSeconds.toFixed(2)} aria-label="Seconds per Gemma call"
                     onChange={(e) => setLlmSeconds(Math.max(0.1, +e.target.value || 0.1))} />
              seconds per call
            </div>
          </div>
        </div>

        <div className="chart-panel">
          <div className="chart-head">
            <p className="chart-title">What each threshold buys</p>
            <div className="segmented" role="group" aria-label="Metric">
              {(Object.keys(METRICS) as MetricKey[]).map((k) => (
                <button key={k} aria-pressed={metric === k} onClick={() => setMetric(k)}>{METRICS[k].label}</button>
              ))}
            </div>
          </div>
          <TradeoffChart curve={data.curve} current={data.current} classifierOnly={data.classifier_only} llmOnly={data.llm_only}
                         metric={m.get} metricLabel={m.label} minutesFor={(x) => minutes(totalMinutes(x))}
                         onPick={(c) => setPolicy({ ...policy, confidence: c })} />
        </div>
      </div>

      <div className="table-scroll">
        <table className="compare">
          <thead>
            <tr>
              <th>Policy</th><th>Sent to Gemma</th><th>Sentiment</th><th>Department</th><th>Urgency</th>
              <th>Churn caught</th><th>Refund flags right</th><th>Time, {data.current.n} tickets</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, color, x]) => (
              <tr key={name} className={name === "Your policy" ? "current" : undefined}>
                <td><span className="swatch" style={{ background: color }} />{name}</td>
                <td>{pct(x.escalation_rate)}</td>
                <td>{pct(x.sentiment)}</td>
                <td>{pct(x.department)}</td>
                <td>{pct(x.urgency)}</td>
                <td>{pct(x.churn.recall)}</td>
                <td>{pct(x.refund.precision)}</td>
                <td>{minutes(totalMinutes(x))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        Accuracy is measured against the labels each ticket was generated from. Gemma also wrote the tickets, so its
        numbers are likely optimistic. Urgency stays near 50% for every policy: the labels themselves are fuzzy.
      </p>
    </>
  );
}
