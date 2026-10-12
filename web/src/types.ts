// Shapes returned by the FastAPI backend (api/main.py).

export interface PolicyParams {
  confidence: number;
  churn_screen: number | null;
  churn_screen_neutral: boolean;
  refund_check: number | null;
}

export interface PRF { precision: number; recall: number; f1: number }

export interface PolicyMetrics {
  policy: PolicyParams & { name: string };
  n: number;
  escalated: number;
  escalation_rate: number;
  reasons: Record<string, number>;
  sentiment: number;
  department: number;
  urgency: number;
  churn: PRF;
  refund: PRF;
  seconds_per_ticket: number;
  total_minutes: number;
  confidence?: number;
}

export interface PolicyResponse {
  current: PolicyMetrics;
  classifier_only: PolicyMetrics;
  llm_only: PolicyMetrics;
  curve: PolicyMetrics[];
}

export interface Summary {
  tickets: number;
  languages: Record<string, number>;
  data_source: "sample" | "local";
  classifier: { name: string; p50_ms: number; p95_ms: number; run: { model_version: string; device: string } };
  llm: { name: string; median_s: number; labelled: number; run: { model: string; prompt_version: string } } | null;
}

export interface CalibrationBin { bin_low: number; bin_high: number; n: number; mean_confidence: number; accuracy: number }
export type Calibration = Record<"sentiment" | "department", { bins: CalibrationBin[]; ece: number; accuracy: number }>;

export interface LanguageRow {
  lang: string; n: number;
  classifier_sentiment: number; llm_sentiment: number | null;
  classifier_department: number; llm_department: number | null;
}

export type Field = "sentiment" | "department" | "urgency" | "churn_risk" | "refund_request";
export type Labels = Record<Field, string | number>;

export interface TicketView {
  id: number; lang: string; channel: string; difficulty: string; text: string;
  escalated: boolean; reasons: string[]; source: "classifier" | "llm";
  truth: Labels; final: Labels; correct: Record<Field, boolean>;
}

export interface TicketDetail extends TicketView {
  classifier: {
    checkpoint: string; detected_lang: string | null; latency_ms: number;
    sentiment: string; sentiment_conf: number; sentiment_probs: Record<string, number>;
    department: string; department_conf: number; department_probs: Record<string, number>;
    urgency_score: number; churn_p: number; refund_p: number;
  };
  llm: (Labels & { summary: string; seconds: number; ok: number }) | null;
}

export interface TriageResult {
  seconds: number;
  classifier: { checkpoint: string; detected_lang: string | null; latency_ms: number; sentiment: string; sentiment_conf: number;
          department: string; department_conf: number; urgency_score: number; churn_p: number; refund_p: number };
  reasons: string[];
  llm: (Labels & { summary?: string; seconds: number; ok: number }) | null;
  final: Labels & { source: string; summary: string | null };
}
