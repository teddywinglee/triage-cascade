import { useState } from "react";
import { api } from "../api";
import type { PolicyParams, TriageResult } from "../types";
import { REASON_TEXT, URGENCY_TEXT, pct } from "../format";

const EXAMPLES = [
  "Oh wonderful, the sync ate my notes again. Really loving this app.",
  "我們團隊在考慮明年要不要續約，最近同步一直出問題，別家好像穩定很多。",
  "¿Pueden devolverme el cargo duplicado de este mes? Gracias.",
];

export function TryIt({ policy }: { policy: PolicyParams }) {
  const [text, setText] = useState(EXAMPLES[0]);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<TriageResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true); setError(null);
    try { setResult(await api.triage(text, policy)); }
    catch (e) { setResult(null); setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };

  return (
    <div className="try">
      <label htmlFor="try-text" className="sr-only" style={{ position: "absolute", left: -9999 }}>Customer message</label>
      <textarea id="try-text" value={text} onChange={(e) => setText(e.target.value)} />
      <div className="try-actions">
        <button className="btn" onClick={run} disabled={busy || text.trim().length < 3}>{busy ? "Triaging…" : "Triage this message"}</button>
        {EXAMPLES.map((ex, i) => (
          <button key={i} className="btn-ghost" onClick={() => setText(ex)}>Example {i + 1}</button>
        ))}
      </div>
      {error && (
        <p className="error">
          {error}. Live triage needs Laya installed with its model weights and an LLM server (such as LM Studio) running; the rest of the dashboard works without them.
        </p>
      )}
      {result && (
        <div className="try-result">
          <div>
            <h3>Laya, {result.classifier.latency_ms.toFixed(0)} ms</h3>
            <dl>
              <dt>Sentiment</dt><dd>{result.classifier.sentiment} ({pct(result.classifier.sentiment_conf)} sure)</dd>
              <dt>Department</dt><dd>{result.classifier.department} ({pct(result.classifier.department_conf)} sure)</dd>
              <dt>Churn score</dt><dd>{result.classifier.churn_p.toFixed(3)}</dd>
              <dt>Refund score</dt><dd>{result.classifier.refund_p.toFixed(2)}</dd>
              <dt>Language</dt><dd>{result.classifier.detected_lang ?? "unknown"} ({result.classifier.checkpoint})</dd>
            </dl>
          </div>
          <div>
            <h3>{result.reasons.length
              ? `Sent to Gemma: ${result.reasons.map((r) => REASON_TEXT[r] ?? r).join(", ")}`
              : "Laya was confident, so Gemma was skipped"}</h3>
            <dl>
              <dt>Sentiment</dt><dd>{String(result.final.sentiment)}</dd>
              <dt>Department</dt><dd>{String(result.final.department)}</dd>
              <dt>Urgency</dt><dd>{URGENCY_TEXT[result.final.urgency as number]}</dd>
              <dt>Churn risk</dt><dd>{result.final.churn_risk ? "Yes" : "No"}</dd>
              <dt>Refund request</dt><dd>{result.final.refund_request ? "Yes" : "No"}</dd>
              {result.final.summary && (<><dt>Summary</dt><dd>{result.final.summary}</dd></>)}
              <dt>Total time</dt><dd>{result.seconds.toFixed(2)} s</dd>
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}
