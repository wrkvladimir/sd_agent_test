import * as React from "react";
import { Download, Plus, Settings, Trash } from "lucide-react";

import { api } from "@/lib/api";
import type { ScenarioDefinition, SgrConvertResponse } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Dropzone } from "@/components/dropzone";
import { InfoBlock } from "@/components/info-block";
import { cn } from "@/lib/utils";

function downloadJson(filename: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function ScenarioCard({ className }: { className?: string }) {
  const [scenarios, setScenarios] = React.useState<ScenarioDefinition[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string>("");

  const [open, setOpen] = React.useState(false);
  const [draftJson, setDraftJson] = React.useState<string>("");
  const [draftNameHint, setDraftNameHint] = React.useState("");
  const [draftText, setDraftText] = React.useState("");
  const [convertBusy, setConvertBusy] = React.useState(false);
  const [uploadBusy, setUploadBusy] = React.useState(false);
  const [convertQuestions, setConvertQuestions] = React.useState<string[]>([]);
  const [convertDiagnostics, setConvertDiagnostics] = React.useState<any>(null);

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const res = await api<ScenarioDefinition[]>("/api/scenarios");
      setScenarios(res || []);
    } catch (e: any) {
      setError(e?.message ? String(e.message) : "Failed to load scenarios");
    } finally {
      setLoading(false);
    }
  }

  React.useEffect(() => {
    refresh();
  }, []);

  async function toggleEnabled(name: string, enabled: boolean) {
    setScenarios((prev) =>
      prev.map((s) => (s.name === name ? { ...s, enabled } : s)),
    );
    try {
      await api<any>(`/api/scenarios/${encodeURIComponent(name)}`, {
        method: "PATCH",
        body: JSON.stringify({ enabled }),
      });
    } catch (e) {
      await refresh();
    }
  }

  async function deleteScenario(name: string) {
    await api<any>(`/api/scenarios/${encodeURIComponent(name)}`, { method: "DELETE" });
    await refresh();
  }

  async function convertToScenario() {
    setConvertBusy(true);
    setConvertQuestions([]);
    setConvertDiagnostics(null);
    try {
      const res = await api<SgrConvertResponse>("/api/sgr/convert", {
        method: "POST",
        body: JSON.stringify({
          text: draftText,
          name_hint: draftNameHint || null,
          strict: true,
          return_diagnostics: true,
        }),
      });
      setDraftJson(JSON.stringify(res.scenario, null, 2));
      setConvertQuestions(res.questions || []);
      setConvertDiagnostics(res.diagnostics || null);
    } catch (e: any) {
      setDraftJson(e?.message ? String(e.message) : "Convert failed");
    } finally {
      setConvertBusy(false);
    }
  }

  async function uploadScenario() {
    setUploadBusy(true);
    try {
      const parsed = JSON.parse(draftJson || "{}");
      await api<any>("/api/scenarios", {
        method: "POST",
        body: JSON.stringify(parsed),
      });
      setOpen(false);
      setDraftJson("");
      setDraftText("");
      setDraftNameHint("");
      setConvertQuestions([]);
      setConvertDiagnostics(null);
      await refresh();
    } catch (e: any) {
      setError(e?.message ? String(e.message) : "Upload failed");
    } finally {
      setUploadBusy(false);
    }
  }

  const busy = loading || convertBusy || uploadBusy;

  return (
    <Card className={cn("relative flex min-h-0 flex-1 flex-col", className)}>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle>Scenario Management</CardTitle>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex min-h-0 flex-1 flex-col p-3 pt-0">
        {error ? <div className="mb-3 text-sm text-destructive">{error}</div> : null}

        <div className="relative flex min-h-0 flex-1 flex-col">
          <ScrollArea className="min-h-0 flex-1 rounded-2xl border border-border bg-background p-2">
            <TooltipProvider delayDuration={250}>
              <div className="flex flex-col gap-3 p-1">
                {scenarios.length === 0 && !loading ? (
                  <div className="p-3 text-sm text-muted-foreground">No scenarios.</div>
                ) : null}
                {scenarios.map((s) => {
                  const enabled = s.enabled !== false;
                  const tooltipSummary = (s.summary || "").trim() || "—";
                  const tooltipAdmin = (s.admin_message || "").trim() || "—";
                  const meta = s.meta || {};
                  return (
                    <div
                      key={s.name}
                      className="rounded-2xl border border-border bg-card p-3 shadow-sm"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="min-w-0 flex-1 cursor-default">
                              <div className="whitespace-normal break-words text-xs font-semibold">
                                {s.name}
                              </div>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent side="top" align="start" sideOffset={10}>
                            <div className="max-w-[420px] space-y-2">
                              <div className="text-sm font-semibold">Summary</div>
                              <div className="text-sm text-muted-foreground">{tooltipSummary}</div>
                              <div className="text-sm font-semibold">Admin</div>
                              <div className="text-sm text-muted-foreground">{tooltipAdmin}</div>
                              <div className="text-sm font-semibold">Meta</div>
                              <pre className="max-h-44 overflow-auto rounded-xl border border-border bg-background p-2 text-xs">
                                {JSON.stringify(meta, null, 2)}
                              </pre>
                            </div>
                          </TooltipContent>
                        </Tooltip>

                        <div className="flex shrink-0 items-center gap-1.5">
                          <div className="flex items-center gap-2">
                            <Checkbox
                              checked={enabled}
                              onCheckedChange={(v) => toggleEnabled(s.name, Boolean(v))}
                              disabled={busy}
                              aria-label="enabled"
                            />
                            <span className={cn("text-xs", enabled ? "text-muted-foreground" : "text-foreground")}>
                              {enabled ? "Enabled" : "Disabled"}
                            </span>
                          </div>

                          <Button
                            variant="outline"
                            size="icon"
                            onClick={() => downloadJson(`${s.name}.json`, s)}
                            disabled={busy}
                            aria-label="download"
                          >
                            <Download className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="icon"
                            onClick={() => deleteScenario(s.name)}
                            disabled={busy}
                            aria-label="delete"
                          >
                            <Trash className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </TooltipProvider>
          </ScrollArea>

          <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
              <Button
                className="absolute bottom-4 right-4 h-11 w-11 rounded-2xl bg-primary/90 hover:bg-primary"
                size="icon"
                aria-label="add scenario"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Add Scenario</DialogTitle>
                <DialogDescription>
                  Convert a description to JSON, or upload a `.json` file.
                </DialogDescription>
              </DialogHeader>

              <div className="grid min-w-0 grid-cols-1 gap-4 md:grid-cols-2">
                <div className="min-w-0 space-y-3">
                  <div className="text-sm font-semibold">From description</div>
                  <Input
                    placeholder="Name hint (optional)"
                    value={draftNameHint}
                    onChange={(e) => setDraftNameHint(e.target.value)}
                  />
                  <Textarea
                    placeholder="Describe the scenario…"
                    value={draftText}
                    onChange={(e) => setDraftText(e.target.value)}
                    className="h-44"
                  />
                  <Button variant="outline" onClick={convertToScenario} disabled={!draftText.trim() || convertBusy}>
                    <Settings className={cn("h-4 w-4", convertBusy && "animate-spin")} />
                    Convert to JSON
                  </Button>

                  {convertQuestions.length ? (
                    <div className="rounded-2xl border border-border bg-background p-4">
                      <div className="text-sm font-semibold">Questions</div>
                      <ul className="mt-2 list-disc space-y-1 pl-4 text-sm text-muted-foreground">
                        {convertQuestions.map((q, idx) => (
                          <li key={idx}>{q}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>

                <div className="min-w-0 space-y-3">
                  <div className="text-sm font-semibold">Scenario JSON</div>
                  <Dropzone
                    accept=".json,application/json"
                    title="Drag & drop a .json scenario"
                    description="Or choose a file from your computer."
                    onFile={async (f) => {
                      const text = await f.text();
                      setDraftJson(text);
                    }}
                  />
                  <Textarea
                    value={draftJson}
                    onChange={(e) => setDraftJson(e.target.value)}
                    placeholder="{ name, code, meta, enabled, summary, admin_message }"
                    className="h-52 font-mono text-xs"
                  />

                  <div className="flex items-center justify-end gap-3">
                    <Button variant="outline" onClick={() => setOpen(false)} disabled={uploadBusy}>
                      Cancel
                    </Button>
                    <Button onClick={uploadScenario} disabled={!draftJson.trim() || uploadBusy}>
                      <Plus className={cn("h-4 w-4", uploadBusy && "animate-spin")} />
                      Upload
                    </Button>
                  </div>
                </div>
              </div>

              <InfoBlock title="Scenario JSON Structure (required)">
                <div className="space-y-2">
                  <div>
                    Top level: `name` (string), `code` (array), `meta` (object), `enabled` (bool).
                  </div>
                  <div>
                    Nodes: `text` (text), `tool` (tool name), `if` (condition + children + else_children), `end`.
                  </div>
                  <div>
                    Optional: `summary` and `admin_message` (shown in tooltips).
                  </div>
                  <pre className="mt-2 rounded-xl border border-border bg-background p-2 text-xs">
{`{
  "name": "example",
  "enabled": true,
  "summary": "What it does (short).",
  "admin_message": "Admin note.",
  "meta": { "apply_only_message_index": 1 },
  "code": [
    { "id": "1", "type": "text", "text": "Instruction…" },
    { "id": "2", "type": "end" }
  ]
}`}
                  </pre>
                </div>
              </InfoBlock>

              {convertDiagnostics ? (
                <InfoBlock title="Diagnostics (read-only)">
                  <pre className="max-h-44 overflow-auto rounded-xl border border-border bg-background p-2 text-xs">
                    {JSON.stringify(convertDiagnostics, null, 2)}
                  </pre>
                </InfoBlock>
              ) : null}
            </DialogContent>
          </Dialog>
        </div>
      </CardContent>
    </Card>
  );
}
