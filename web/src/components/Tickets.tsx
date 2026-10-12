import { useEffect, useState } from "react";
import { api } from "../api";
import type { Field, PolicyParams, TicketDetail, TicketView } from "../types";
import { LANG_NAMES, REASON_TEXT, URGENCY_TEXT, pct } from "../format";

const FIELD_NAMES: Record<Field, string> = {
  sentiment: "Sentiment", department: "Department", urgency: "Urgency",
  churn_risk: "Churn risk", refund_request: "Refund request",
};
const FIELDS = Object.keys(FIELD_NAMES) as Field[];

function show(field: Field, v: string | number) {
  if (field === "urgency") return URGENCY_TEXT[v as number] ?? String(v);
  if (field === "churn_risk" || field === "refund_request") return v ? "Yes" : "No";
  return String(v);
}

export function Tickets({ policy }: { policy: PolicyParams }) {
  const [filters, setFilters] = useState({ lang: "", escalated: "", wrong: "", difficulty: "" });
  const [list, setList] = useState<TicketView[] | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [detail, setDetail] = useState<TicketDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.tickets(policy, filters).then((l) => {
      if (!live) return;
      setList(l);
      setError(null);
      if (l.length && (selected === null || !l.some((t) => t.id === selected))) setSelected(l[0].id);
    }).catch((e) => live && setError(String(e.message ?? e)));
    return () => { live = false; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [policy, filters]);

  useEffect(() => {
    if (selected === null) { setDetail(null); return; }
    let live = true;
    api.ticket(selected, policy).then((d) => live && setDetail(d)).catch(() => live && setDetail(null));
    return () => { live = false; };
  }, [selected, policy]);

  const set = (k: keyof typeof filters) => (e: React.ChangeEvent<HTMLSelectElement>) => setFilters({ ...filters, [k]: e.target.value });

  return (
    <>
      <div className="filters">
        <label>Language
          <select value={filters.lang} onChange={set("lang")}>
            <option value="">All languages</option>
            {Object.entries(LANG_NAMES).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </label>
        <label>Handled by
          <select value={filters.escalated} onChange={set("escalated")}>
            <option value="">Laya or Gemma</option>
            <option value="false">Laya only</option>
            <option value="true">Sent to Gemma</option>
          </select>
        </label>
        <label>Got wrong
          <select value={filters.wrong} onChange={set("wrong")}>
            <option value="">Any result</option>
            {FIELDS.map((f) => <option key={f} value={f}>{FIELD_NAMES[f]}</option>)}
          </select>
        </label>
        <label>Writing style
          <select value={filters.difficulty} onChange={set("difficulty")}>
            <option value="">Any style</option>
            {["plain", "sarcasm", "mixed", "implicit_churn", "code_switch", "informal", "long"].map((d) =>
              <option key={d} value={d}>{d.replace("_", " ")}</option>)}
          </select>
        </label>
        {list && <span className="result-count">{list.length} tickets</span>}
      </div>

      {error && <p className="error">Couldn't load tickets: {error}</p>}

      <div className="tickets">
        <ul className="ticket-list" aria-label="Tickets">
          {list?.length === 0 && <li className="empty">No tickets match these filters. Try clearing one.</li>}
          {list?.map((t) => {
            const wrong = FIELDS.filter((f) => !t.correct[f]);
            return (
              <li key={t.id}>
                <button aria-current={t.id === selected} onClick={() => setSelected(t.id)}>
                  <span className="ticket-text">{t.text}</span>
                  <span className={`route-pill ${t.source}`}>{t.source === "llm" ? "Gemma" : "Laya"}</span>
                  <span className="ticket-meta">
                    <span>#{t.id}</span>
                    <span>{LANG_NAMES[t.lang] ?? t.lang}</span>
                    <span>{t.difficulty.replace("_", " ")}</span>
                    {wrong.length > 0
                      ? <span className="wrong">Wrong: {wrong.map((f) => FIELD_NAMES[f].toLowerCase()).join(", ")}</span>
                      : <span>All correct</span>}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>

        <div className="detail" aria-live="polite">
          {!detail ? <p className="empty">Pick a ticket to see how it was handled.</p> : (
            <>
              <blockquote>{detail.text}</blockquote>
              <div className="reasons">
                {detail.escalated
                  ? detail.reasons.map((r) => <span key={r} className="route-pill llm">{REASON_TEXT[r] ?? r}</span>)
                  : <span className="route-pill classifier">Laya was confident: no escalation</span>}
              </div>
              {detail.escalated && detail.llm?.summary && <p className="summary-line">Gemma's summary: {detail.llm.summary}</p>}

              <h3>Answers</h3>
              <div className="table-scroll"><table className="labels">
                <thead><tr><th></th><th>Correct</th><th>Laya</th><th>{detail.escalated ? "Gemma" : "Gemma (not asked)"}</th><th>Final</th></tr></thead>
                <tbody>
                  {FIELDS.map((f) => {
                    // Laya's raw output: a label for the choice questions, a score/probability otherwise.
                    const classifierVal =
                      f === "sentiment" ? detail.classifier.sentiment
                      : f === "department" ? detail.classifier.department
                      : f === "urgency" ? `score ${detail.classifier.urgency_score.toFixed(2)}`
                      : f === "churn_risk" ? `p = ${detail.classifier.churn_p.toFixed(3)}`
                      : `p = ${detail.classifier.refund_p.toFixed(2)}`;
                    return (
                      <tr key={f}>
                        <th scope="row">{FIELD_NAMES[f]}</th>
                        <td>{show(f, detail.truth[f])}</td>
                        <td>{classifierVal}</td>
                        <td>{detail.llm?.ok ? show(f, detail.llm[f]) : "–"}</td>
                        <td className={detail.correct[f] ? "mark-ok" : "mark-bad"}>
                          {show(f, detail.final[f])} {detail.correct[f] ? "✓" : "✗"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table></div>

              <h3>Laya's confidence</h3>
              {(["sentiment", "department"] as const).map((q) => (
                <div key={q} style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: "0.8125rem", color: "var(--ink-2)", marginBottom: 2 }}>{FIELD_NAMES[q]}</div>
                  {Object.entries(detail.classifier[`${q}_probs`]).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                    <div className="prob-row" key={k}>
                      <span>{k}</span>
                      <span className="prob-track"><span className="prob-fill" style={{ width: `${v * 100}%`, display: "block" }} /></span>
                      <span className="v">{pct(v)}</span>
                    </div>
                  ))}
                </div>
              ))}
              <p className="note">
                Laya: {detail.classifier.checkpoint} checkpoint, {detail.classifier.latency_ms.toFixed(0)} ms, detected language {detail.classifier.detected_lang ?? "unknown"}.
                {detail.llm?.ok ? ` Gemma: ${detail.llm.seconds.toFixed(1)} s.` : ""}
              </p>
            </>
          )}
        </div>
      </div>
    </>
  );
}
