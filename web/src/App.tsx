import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type { Calibration, LanguageRow, PolicyParams, PolicyResponse, Summary } from "./types";
import { minutes, pct } from "./format";
import { PolicyLab } from "./components/PolicyLab";
import { Tickets } from "./components/Tickets";
import { TryIt } from "./components/TryIt";
import { ReliabilityChart } from "./charts/ReliabilityChart";
import { DumbbellChart } from "./charts/DumbbellChart";

const DEFAULT_POLICY: PolicyParams = { confidence: 0.9, churn_screen: 0.005, churn_screen_neutral: true, refund_check: 0.5 };

function useDebounced<T>(value: T, ms: number) {
  const [v, setV] = useState(value);
  useEffect(() => { const t = setTimeout(() => setV(value), ms); return () => clearTimeout(t); }, [value, ms]);
  return v;
}

type Theme = "auto" | "light" | "dark";

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [cal, setCal] = useState<Calibration | null>(null);
  const [langs, setLangs] = useState<LanguageRow[] | null>(null);
  const [langField, setLangField] = useState<"department" | "sentiment">("department");
  const [policy, setPolicy] = useState<PolicyParams>(DEFAULT_POLICY);
  const debounced = useDebounced(policy, 120);
  const [result, setResult] = useState<PolicyResponse | null>(null);
  const [llmSeconds, setLlmSeconds] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [theme, setTheme] = useState<Theme>("auto");

  useEffect(() => {
    Promise.all([api.summary(), api.calibration(), api.languages()])
      .then(([s, c, l]) => { setSummary(s); setCal(c); setLangs(l); setLlmSeconds((g) => g ?? s.llm?.median_s ?? 1.3); })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    let live = true;
    api.policy(debounced).then((r) => live && setResult(r)).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [debounced]);

  useEffect(() => {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const classifierSeconds = (summary?.classifier.p50_ms ?? 20) / 1000;
  const g = llmSeconds ?? 1.3;
  const live = useMemo(() => {
    if (!result) return null;
    const c = result.current;
    const policyMin = (c.n * classifierSeconds + c.escalated * g) / 60;
    const llmMin = (c.n * g) / 60;
    return { c, policyMin, llmMin, saved: 1 - policyMin / llmMin };
  }, [result, classifierSeconds, g]);

  if (error && !summary) {
    return (
      <main className="page">
        <p className="wordmark">Triage Cascade</p>
        <div className="banner" style={{ marginTop: 32 }}>
          The dashboard couldn't reach the API ({error}). Start it from the project folder with
          <code> uvicorn api.main:app --port 8000</code>, then reload this page.
        </div>
      </main>
    );
  }

  return (
    <main className="page">
      <header className="masthead">
        <p className="wordmark">Triage Cascade</p>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {summary && (
            <p className="masthead-meta">
              {summary.tickets} synthetic tickets in {Object.keys(summary.languages).length} languages for the fictional Nimbus Notes app
              {summary.data_source === "sample" && " (committed sample results)"}
            </p>
          )}
          <button className="theme-toggle" onClick={() => setTheme(theme === "auto" ? "dark" : theme === "dark" ? "light" : "auto")}>
            Theme: {theme}
          </button>
        </div>
      </header>

      <h1 className="thesis">
        Laya reads a ticket in <span className="num classifier">{summary ? `${summary.classifier.p50_ms.toFixed(0)} ms` : "…"}</span>.
        Gemma takes <span className="num llm">{summary?.llm ? `${summary.llm.median_s.toFixed(1)} s` : "…"}</span>.
        Most tickets don't need the wait.
      </h1>
      {live ? (
        <p className="thesis-live">
          With your policy, <strong>{pct(live.c.escalation_rate)}</strong> of tickets go to Gemma.
          Department accuracy is <strong>{pct(live.c.department)}</strong> (Gemma alone: {pct(result!.llm_only.department)}),
          and the batch takes <strong>{minutes(live.policyMin)}</strong> instead of {minutes(live.llmMin)},
          <strong> {pct(live.saved)}</strong> less time.
        </p>
      ) : <p className="thesis-live loading">Loading results…</p>}
      <p className="thesis-sub">
        Laya is a non-generative decision model: an encoder that scores fixed answer options in one pass and returns
        a probability for each. Gemma 4 12B is a generative LLM running locally. Laya handles what it's sure about
        and hands the rest to Gemma. Adjust the policy below to trade speed for quality.
      </p>

      <section aria-labelledby="lab-h">
        <h2 id="lab-h">Choose which tickets go to Gemma</h2>
        <p className="lede">
          Each point on the curve is one confidence threshold; click one to use it. With a double-check switched on,
          even the lowest threshold sends some tickets to Gemma, so the curve starts to the right of "Laya only".
        </p>
        {result && summary ? (
          <PolicyLab policy={policy} setPolicy={setPolicy} data={result} classifierSeconds={classifierSeconds}
                     llmSeconds={g} setLlmSeconds={setLlmSeconds}
                     measuredLlmSeconds={summary.llm?.median_s ?? 1.3} />
        ) : <p className="loading">Loading policy results…</p>}
      </section>

      <section aria-labelledby="cal-h">
        <h2 id="cal-h">Can Laya's confidence be trusted?</h2>
        <p className="lede">
          Partly. Laya is overconfident: answers given 70–80% confidence are right only about half the time.
          But higher confidence still means more often right, which is all a threshold needs.
        </p>
        {cal && (
          <div className="two-up">
            {(["sentiment", "department"] as const).map((q) => (
              <div className="chart-panel" key={q}>
                <div className="chart-head">
                  <p className="chart-title">{q === "sentiment" ? "Sentiment" : "Department"}</p>
                  <span className="masthead-meta">accuracy {pct(cal[q].accuracy)}, calibration error {cal[q].ece.toFixed(2)}</span>
                </div>
                <ReliabilityChart bins={cal[q].bins} title={q} />
              </div>
            ))}
          </div>
        )}
      </section>

      <section aria-labelledby="lang-h">
        <h2 id="lang-h">How each language fares</h2>
        <p className="lede">Laya's weakest languages are where escalation earns the most. Each row compares the two models on the same 50 tickets.</p>
        <div className="chart-panel">
          <div className="chart-head">
            <div className="legend">
              <span><i style={{ background: "var(--classifier)" }} />Laya (classifier)</span>
              <span><i style={{ background: "var(--llm)" }} />Gemma (LLM)</span>
            </div>
            <div className="segmented" role="group" aria-label="Question">
              {(["department", "sentiment"] as const).map((f) => (
                <button key={f} aria-pressed={langField === f} onClick={() => setLangField(f)}>
                  {f === "department" ? "Department" : "Sentiment"}
                </button>
              ))}
            </div>
          </div>
          {langs && <DumbbellChart rows={langs} field={langField} />}
        </div>
      </section>

      <section aria-labelledby="tickets-h">
        <h2 id="tickets-h">Inspect the tickets</h2>
        <p className="lede">See how each ticket was routed under your policy, and where the final answer went wrong.</p>
        <Tickets policy={debounced} />
      </section>

      <section aria-labelledby="try-h">
        <h2 id="try-h">Triage a new message</h2>
        <p className="lede">Runs the full LangGraph cascade live: Laya first, then Gemma only if your policy says so.</p>
        <TryIt policy={debounced} />
      </section>
    </main>
  );
}
