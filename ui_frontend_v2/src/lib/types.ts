export type MessageRole = "user" | "assistant" | "system";

export type HistoryItem = {
  role: MessageRole;
  content: string;
  timestamp?: string;
};

export type HistoryResponse = {
  conversation_id: string;
  history: HistoryItem[];
};

export type SummaryResponse = {
  conversation_id: string;
  summary: string;
};

export type ChatJob = {
  job_id: string;
  status: "pending" | "done" | "error";
  result?: any;
  error?: string | null;
};

export type RuntimeConfig = {
  OPENAI_API_KEY: string;
  RERANKER_ENABLED: boolean;
  SEARCH_TOP_K: number;
  SEARCH_LIMIT: number;
  SEARCH_SCORE_THRESHOLD: number;
  CHUNK_MAX_LENGTH: number;
  CHUNK_OVERLAP: number;
  AGENT_PIPELINE_VERSION: string;
};

export type RuntimeConfigResponse = {
  values: RuntimeConfig;
  overrides: Partial<RuntimeConfig>;
  defaults: RuntimeConfig;
};

export type ChatConfig = {
  default_pipeline_version: string;
  supported_pipeline_versions: string[];
};

export type ScenarioNode =
  | { id: string; type: "text"; text: string }
  | { id: string; type: "tool"; tool: string }
  | {
      id: string;
      type: "if";
      condition: string;
      children: ScenarioNode[];
      else_children: ScenarioNode[];
    }
  | { id: string; type: "end" };

export type ScenarioDefinition = {
  name: string;
  code: ScenarioNode[];
  meta: Record<string, any>;
  enabled?: boolean;
  summary?: string | null;
  admin_message?: string | null;
};

export type SgrConvertResponse = {
  scenario: ScenarioDefinition;
  diagnostics: Record<string, any>;
  questions: string[];
};

