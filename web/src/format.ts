export const pct = (x: number | null | undefined, digits = 0) =>
  x === null || x === undefined ? "–" : `${(x * 100).toFixed(digits)}%`;

export const LANG_NAMES: Record<string, string> = {
  en: "English", zh: "Chinese (Trad.)", ja: "Japanese", ko: "Korean", es: "Spanish",
  pt: "Portuguese", fr: "French", de: "German", hi: "Hindi",
};

export const REASON_TEXT: Record<string, string> = {
  low_confidence: "Laya unsure",
  churn_screen: "Possible churn",
  refund_check: "Refund claim",
  always: "Always escalate",
};

export const URGENCY_TEXT = ["Not urgent", "Soon", "Blocking"];

export function minutes(m: number) {
  if (m < 1) return `${Math.round(m * 60)} s`;
  if (m < 120) return `${m.toFixed(1)} min`;
  return `${(m / 60).toFixed(1)} h`;
}
