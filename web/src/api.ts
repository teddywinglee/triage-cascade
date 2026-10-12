import type { Calibration, LanguageRow, PolicyParams, PolicyResponse, Summary, TicketDetail, TicketView, TriageResult } from "./types";

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, body === undefined ? undefined : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  summary: () => call<Summary>("/api/summary"),
  policy: (p: PolicyParams) => call<PolicyResponse>("/api/policy", p),
  calibration: () => call<Calibration>("/api/calibration"),
  languages: () => call<LanguageRow[]>("/api/languages"),
  tickets: (p: PolicyParams, filters: Record<string, string>) => {
    const qs = new URLSearchParams(Object.entries(filters).filter(([, v]) => v !== "")).toString();
    return call<TicketView[]>(`/api/tickets${qs ? `?${qs}` : ""}`, p);
  },
  ticket: (id: number, p: PolicyParams) => call<TicketDetail>(`/api/tickets/${id}`, p),
  triage: (text: string, policy: PolicyParams) => call<TriageResult>("/api/triage", { text, policy }),
};
