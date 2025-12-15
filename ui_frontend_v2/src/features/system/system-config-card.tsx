import * as React from "react";
import { Database, Play, Settings } from "lucide-react";

import { api } from "@/lib/api";
import type { ChatConfig, RuntimeConfigResponse } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const VERSION_DESCRIPTIONS: Record<string, string> = {
  "0.1": "One-Shot Prompting — scenarios are processed by simple functions, the full context is consolidated into a single request to the LLM. Fast, cheap, not scalable for many scenarios, requires programming of certain scenario trigger conditions.",
  "1.0": "LangGraph pipeline and dedicated subgraph with tools and Map-Reduce tool for asynchronous parsing of scenarios and conditions. At least 4 LLM calls. Slower, more expensive, higher quality, scalable.",
};

export function SystemConfigCard({ className }: { className?: string }) {
  const [runtime, setRuntime] = React.useState<RuntimeConfigResponse | null>(null);
  const [chatCfg, setChatCfg] = React.useState<ChatConfig | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const [values, setValues] = React.useState<any>(null);
  const [graphToast, setGraphToast] = React.useState<string>("");

  async function load() {
    setBusy(true);
    setError("");
    try {
      const [r, c] = await Promise.all([
        api<RuntimeConfigResponse>("/api/runtime-config"),
        api<ChatConfig>("/api/config"),
      ]);
      setRuntime(r);
      setChatCfg(c);
      setValues(r.values);
    } catch (e: any) {
      setError(e?.message ? String(e.message) : "Failed to load config");
    } finally {
      setBusy(false);
    }
  }

  React.useEffect(() => {
    load();
  }, []);

  async function patch(patchValues: Record<string, any>) {
    setBusy(true);
    setError("");
    try {
      const res = await api<RuntimeConfigResponse>("/api/runtime-config", {
        method: "PATCH",
        body: JSON.stringify(patchValues),
      });
      setRuntime(res);
      setValues(res.values);
    } catch (e: any) {
      setError(e?.message ? String(e.message) : "Save failed");
    } finally {
      setBusy(false);
    }
  }

  // Auto-apply config changes with debounce (no explicit Save button).
  React.useEffect(() => {
    if (!runtime?.values || !values) return;
    if (busy) return;

    const keys = [
      "OPENAI_API_KEY",
      "RERANKER_ENABLED",
      "SEARCH_TOP_K",
      "SEARCH_LIMIT",
      "SEARCH_SCORE_THRESHOLD",
      "CHUNK_MAX_LENGTH",
      "CHUNK_OVERLAP",
      "AGENT_PIPELINE_VERSION",
    ] as const;

    const diff: Record<string, any> = {};
    for (const k of keys) {
      if (values[k] !== runtime.values[k]) diff[k] = values[k];
    }
    if (Object.keys(diff).length === 0) return;

    const t = window.setTimeout(() => {
      patch(diff).catch(() => {});
    }, 450);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [values, runtime?.values]);

  const supportedVersions = chatCfg?.supported_pipeline_versions || ["0.1", "1.0"];
  const currentVersion = values?.AGENT_PIPELINE_VERSION || runtime?.values.AGENT_PIPELINE_VERSION || "0.1";
  const versionDescription = VERSION_DESCRIPTIONS[currentVersion] || "—";

  return (
    <Card className={cn("flex min-h-0 flex-1 flex-col", className)}>
      <CardHeader>
        <CardTitle>System Config</CardTitle>
      </CardHeader>

      <CardContent className="flex min-h-0 flex-1 flex-col p-3 pt-0">
        {error ? <div className="mb-3 text-sm text-destructive">{error}</div> : null}

        <ScrollArea className="min-h-0 flex-1 rounded-2xl border border-border bg-background p-2">
          <div className="grid grid-cols-1 gap-4 p-1">
          <div className="relative rounded-2xl border border-border bg-card p-3 shadow-sm">
            <div className="text-sm font-semibold">Links</div>
            <div className="mt-3 flex flex-wrap gap-3">
              <a
                href="http://localhost:8001/docs"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-xl border border-border bg-background px-2.5 py-2 text-xs hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <Database className="h-4 w-4" />
                ingest_and_retrieval
              </a>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-xl border border-border bg-background px-2.5 py-2 text-xs hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <Settings className="h-4 w-4" />
                chat_app
              </a>
              <button
                type="button"
                onClick={() => {
                  if (String(currentVersion) !== "1.0") {
                    setGraphToast("Graph is not supported on v0.1");
                    window.setTimeout(() => setGraphToast(""), 1800);
                    return;
                  }
                  window.open("http://localhost:8080/api/graph?format=png&xray=1", "_blank", "noopener,noreferrer");
                }}
                className="inline-flex items-center gap-2 rounded-xl border border-border bg-background px-2.5 py-2 text-xs hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <Play className="h-4 w-4" />
                graph
              </button>
            </div>
            {graphToast ? (
              <div className="pointer-events-none absolute right-3 top-3 rounded-xl border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-md animate-in fade-in">
                {graphToast}
              </div>
            ) : null}
          </div>

          <div className="rounded-2xl border border-border bg-card p-3 shadow-sm">
            <div className="flex items-center justify-between gap-3">
              <div className="text-sm font-semibold">Default Pipeline Version</div>
              <Select
                value={currentVersion}
                onValueChange={(v) => {
                  setValues((prev: any) => ({ ...(prev || {}), AGENT_PIPELINE_VERSION: v }));
                }}
              >
                <SelectTrigger className="h-8 w-[110px]">
                  <SelectValue placeholder="Select version" />
                </SelectTrigger>
                <SelectContent>
                  {supportedVersions.map((v) => (
                    <SelectItem key={v} value={v}>
                      {v}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Textarea readOnly value={versionDescription} className="mt-3 min-h-[56px] resize-none" />
          </div>

          <div className="rounded-2xl border border-border bg-card p-3 shadow-sm">
            <div className="text-sm font-semibold">Runtime Fields</div>

            <div className="mt-3 grid grid-cols-1 gap-3">
              <div>
                <div className="mb-1 text-[10px] text-muted-foreground">OPENAI_API_KEY</div>
                <Input
                  type="password"
                  value={values?.OPENAI_API_KEY ?? ""}
                  onChange={(e) => {
                    setValues((prev: any) => ({ ...(prev || {}), OPENAI_API_KEY: e.target.value }));
                  }}
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="mb-1 text-[10px] text-muted-foreground">SEARCH_TOP_K</div>
                  <Input
                    inputMode="numeric"
                    value={String(values?.SEARCH_TOP_K ?? "")}
                    onChange={(e) => {
                      setValues((prev: any) => ({
                        ...(prev || {}),
                        SEARCH_TOP_K: Number(e.target.value || 0),
                      }));
                    }}
                  />
                </div>
                <div>
                  <div className="mb-1 text-[10px] text-muted-foreground">SEARCH_LIMIT</div>
                  <Input
                    inputMode="numeric"
                    value={String(values?.SEARCH_LIMIT ?? "")}
                    onChange={(e) => {
                      setValues((prev: any) => ({
                        ...(prev || {}),
                        SEARCH_LIMIT: Number(e.target.value || 0),
                      }));
                    }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="mb-1 text-[10px] text-muted-foreground">SEARCH_SCORE_THRESHOLD</div>
                  <Input
                    inputMode="decimal"
                    value={String(values?.SEARCH_SCORE_THRESHOLD ?? "")}
                    onChange={(e) => {
                      setValues((prev: any) => ({
                        ...(prev || {}),
                        SEARCH_SCORE_THRESHOLD: Number(e.target.value || 0),
                      }));
                    }}
                  />
                </div>
                <div>
                  <div className="mb-1 text-[10px] text-muted-foreground">RERANKER_ENABLED</div>
                  <div className="flex h-9 items-center justify-between rounded-xl border border-input bg-background px-3">
                    <span className="text-xs text-muted-foreground">Enable rerank</span>
                    <Checkbox
                      checked={Boolean(values?.RERANKER_ENABLED)}
                      onCheckedChange={(v) => {
                        setValues((prev: any) => ({ ...(prev || {}), RERANKER_ENABLED: Boolean(v) }));
                      }}
                    />
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <div className="mb-1 text-[10px] text-muted-foreground">CHUNK_MAX_LENGTH</div>
                  <Input
                    inputMode="numeric"
                    value={String(values?.CHUNK_MAX_LENGTH ?? "")}
                    onChange={(e) => {
                      setValues((prev: any) => ({
                        ...(prev || {}),
                        CHUNK_MAX_LENGTH: Number(e.target.value || 0),
                      }));
                    }}
                  />
                </div>
                <div>
                  <div className="mb-1 text-[10px] text-muted-foreground">CHUNK_OVERLAP</div>
                  <Input
                    inputMode="numeric"
                    value={String(values?.CHUNK_OVERLAP ?? "")}
                    onChange={(e) => {
                      setValues((prev: any) => ({
                        ...(prev || {}),
                        CHUNK_OVERLAP: Number(e.target.value || 0),
                      }));
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
