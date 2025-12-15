import * as React from "react";
import { Plus, RefreshCw, Play } from "lucide-react";

import { api } from "@/lib/api";
import type { ChatJob, HistoryItem, HistoryResponse, RuntimeConfigResponse, SummaryResponse } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { ConversationIdField } from "@/features/chat/conversation-id-field";

type Stage = "search" | "prompt" | "generate";

type Instance = {
  id: string;
  conversationId: string;
  draft: string;
  messages: HistoryItem[];
  sending: boolean;
  stage: Stage | null;
  summary: string;
  summaryStatus: "idle" | "waiting" | "ready" | "error";
  error: string;
};

function makeConversationId(): string {
  const id =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? (crypto.randomUUID as any)()
      : Math.random().toString(36).slice(2);
  return `conv-${id}`;
}

async function pollJob(jobId: string): Promise<ChatJob> {
  while (true) {
    const res = await api<ChatJob>(`/api/chat/poll?job_id=${encodeURIComponent(jobId)}`);
    if (res.status === "pending") {
      await new Promise((r) => setTimeout(r, 700));
      continue;
    }
    return res;
  }
}

export function ChatInstances({ className }: { className?: string }) {
  const [knownConversations, setKnownConversations] = React.useState<string[]>([]);
  const [instances, setInstances] = React.useState<Instance[]>(() => [
    {
      id: "inst-1",
      conversationId: makeConversationId(),
      draft: "",
      messages: [],
      sending: false,
      stage: null,
      summary: "",
      summaryStatus: "idle",
      error: "",
    },
  ]);

  const loadControllers = React.useRef<Map<string, AbortController>>(new Map());

  async function refreshConversations() {
    try {
      const res = await api<{ conversations: string[] }>("/api/chat/conversations");
      setKnownConversations(res.conversations || []);
    } catch {
      setKnownConversations([]);
    }
  }

  React.useEffect(() => {
    refreshConversations();
  }, []);

  async function loadHistoryAndSummary(instanceId: string, conversationId: string) {
    const prev = loadControllers.current.get(instanceId);
    prev?.abort();
    const controller = new AbortController();
    loadControllers.current.set(instanceId, controller);
    const signal = controller.signal;

    setInstances((items) =>
      items.map((i) =>
        i.id === instanceId
          ? { ...i, error: "", messages: [], summary: "", summaryStatus: "waiting" }
          : i,
      ),
    );

    let history: HistoryItem[] = [];
    try {
      const res = await api<HistoryResponse>(
        `/api/chat/history?conversation_id=${encodeURIComponent(conversationId)}`,
        { signal },
      );
      history = (res.history || []) as HistoryItem[];
    } catch {
      // if empty/error: keep empty per spec
      history = [];
    }

    if (signal.aborted) return;

    setInstances((items) =>
      items.map((i) => (i.id === instanceId ? { ...i, messages: history } : i)),
    );

    // Summary: always try to fetch; if empty, keep waiting and poll up to 60s.
    const start = Date.now();
    while (!signal.aborted && Date.now() - start < 60_000) {
      try {
        const res = await api<SummaryResponse>(
          `/api/chat/summary?conversation_id=${encodeURIComponent(conversationId)}`,
          { signal },
        );
        const text = (res.summary || "").trim();
        setInstances((items) =>
          items.map((i) =>
            i.id === instanceId
              ? {
                  ...i,
                  summary: text,
                  summaryStatus: text ? "ready" : "waiting",
                }
              : i,
            ),
        );
        if (text) break;
      } catch {
        // ignore
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  async function pollSummaryOnly(instanceId: string, conversationId: string, baselineSummary: string) {
    const controller = new AbortController();
    const signal = controller.signal;
    const start = Date.now();
    const baseline = (baselineSummary || "").trim();
    let candidate = "";
    let stable = 0;
    while (!signal.aborted && Date.now() - start < 60_000) {
      try {
        const res = await api<SummaryResponse>(
          `/api/chat/summary?conversation_id=${encodeURIComponent(conversationId)}`,
          { signal },
        );
        const text = (res.summary || "").trim();
        setInstances((items) =>
          items.map((i) =>
            i.id === instanceId
              ? {
                  ...i,
                  summary: text,
                  summaryStatus: text ? "ready" : "waiting",
                }
              : i,
          ),
        );
        if (!text) {
          stable = 0;
          candidate = "";
        } else if (baseline && text === baseline) {
          // Still the old summary: keep waiting.
          stable = 0;
          candidate = "";
        } else if (!candidate) {
          // First new summary value.
          candidate = text;
          stable = 0;
        } else if (text === candidate) {
          // New summary stabilized (2 identical polls after change).
          stable += 1;
          if (stable >= 1) return;
        } else {
          // Summary changed again; treat it as a new candidate.
          candidate = text;
          stable = 0;
        }
      } catch {
        // ignore
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  // Auto-load when conversation_id changes (debounced).
  React.useEffect(() => {
    const timers: number[] = [];
    for (const inst of instances) {
      const id = inst.conversationId.trim();
      if (!id) continue;
      const t = window.setTimeout(() => {
        loadHistoryAndSummary(inst.id, id).catch(() => {});
      }, 350);
      timers.push(t);
    }
    return () => {
      timers.forEach((t) => window.clearTimeout(t));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [instances.map((i) => `${i.id}:${i.conversationId}`).join("|")]);

  function addInstance() {
    setInstances((prev) => [
      ...prev,
      {
        id: `inst-${prev.length + 1}-${Date.now()}`,
        conversationId: makeConversationId(),
        draft: "",
        messages: [],
        sending: false,
        stage: null,
        summary: "",
        summaryStatus: "idle",
        error: "",
      },
    ]);
  }

  function removeInstance(id: string) {
    setInstances((prev) => prev.filter((i) => i.id !== id));
  }

  function setStage(instanceId: string, stage: Stage | null) {
    setInstances((prev) => prev.map((i) => (i.id === instanceId ? { ...i, stage } : i)));
  }

  async function sendMessage(instanceId: string) {
    const inst = instances.find((i) => i.id === instanceId);
    if (!inst) return;
    const message = (inst.draft || "").trim();
    if (!message) return;

    const conversationId = (inst.conversationId || "").trim() || makeConversationId();
    const baselineSummary = (inst.summary || "").trim();

    setInstances((prev) =>
      prev.map((i) =>
        i.id === instanceId
          ? {
              ...i,
              conversationId,
              draft: "",
              error: "",
              sending: true,
              stage: "search",
              messages: [...(i.messages || []), { role: "user", content: message }],
              summaryStatus: "waiting",
            }
          : i,
      ),
    );

    const stageTimer = window.setInterval(() => {
      setInstances((prev) =>
        prev.map((i) => {
          if (i.id !== instanceId) return i;
          if (!i.sending) return i;
          if (i.stage === "search") return { ...i, stage: "prompt" };
          if (i.stage === "prompt") return { ...i, stage: "generate" };
          return i;
        }),
      );
    }, 900);

    try {
      // Optional: explicitly pass the current global pipeline version (from System Config),
      // without showing any per-instance selector in the UI.
      let pipelineVersion: string | undefined;
      try {
        const runtime = await api<RuntimeConfigResponse>("/api/runtime-config");
        pipelineVersion = String(runtime?.values?.AGENT_PIPELINE_VERSION || "").trim() || undefined;
      } catch {
        pipelineVersion = undefined;
      }

      const start = await api<{ job_id: string }>("/api/chat/send", {
        method: "POST",
        body: JSON.stringify({
          conversation_id: conversationId,
          message,
          pipeline_version: pipelineVersion,
        }),
      });

      const job = await pollJob(start.job_id);
      if (job.status === "error") throw new Error(job.error || "chat error");
      const answer = String(job.result?.answer || "");

      setInstances((prev) =>
        prev.map((i) =>
          i.id === instanceId
            ? {
                ...i,
                sending: false,
                stage: null,
                messages: [...(i.messages || []), { role: "assistant", content: answer }],
              }
            : i,
        ),
      );

      window.setTimeout(() => {
        pollSummaryOnly(instanceId, conversationId, baselineSummary).catch(() => {});
      }, 300);
    } catch (e: any) {
      setInstances((prev) =>
        prev.map((i) =>
          i.id === instanceId
            ? {
                ...i,
                sending: false,
                stage: null,
                error: e?.message ? String(e.message) : "Send failed",
              }
            : i,
        ),
      );
    } finally {
      window.clearInterval(stageTimer);
      setStage(instanceId, null);
    }
  }

  return (
    <div className={cn("h-full min-h-0 overflow-hidden", className)}>
      <div className="flex h-full min-h-0 flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-base font-semibold">Chat Instances</div>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="outline" onClick={refreshConversations}>
              <RefreshCw className="h-4 w-4" />
              Refresh IDs
            </Button>
            <Button onClick={addInstance}>
              <Plus className="h-4 w-4" />
              Add
            </Button>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-hidden">
          <div className="h-full overflow-x-auto">
            <div className="flex h-full min-h-0 items-stretch gap-4 pb-1">
              {instances.map((inst) => (
                <ChatInstance
                  key={inst.id}
                  instance={inst}
                  knownConversations={knownConversations}
                  onRemove={() => removeInstance(inst.id)}
                  onChangeConversation={(conversationId) =>
                    setInstances((prev) =>
                      prev.map((i) => (i.id === inst.id ? { ...i, conversationId } : i)),
                    )
                  }
                  onChangeDraft={(draft) =>
                    setInstances((prev) =>
                      prev.map((i) => (i.id === inst.id ? { ...i, draft } : i)),
                    )
                  }
                  onNewConversation={() => {
                    const id = makeConversationId();
                    setInstances((prev) =>
                      prev.map((i) =>
                        i.id === inst.id
                          ? { ...i, conversationId: id, messages: [], summary: "", summaryStatus: "idle", error: "" }
                          : i,
                      ),
                    );
                  }}
                  onSend={() => sendMessage(inst.id)}
                />
              ))}

              <div className="flex w-24 shrink-0 items-center justify-center">
                <Button variant="outline" size="icon" onClick={addInstance} aria-label="add instance">
                  <Plus className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StageInline({ stage }: { stage: Stage }) {
  const entries: Array<{ key: Stage; label: string }> = [
    { key: "search", label: "vector storage search" },
    { key: "prompt", label: "prompt building" },
    { key: "generate", label: "waiting for response generation" },
  ];
  return (
    <div className="mt-1 text-xs text-muted-foreground">
      {entries.map((e, idx) => (
        <span key={e.key}>
          <span className={cn(e.key === stage ? "text-foreground" : "")}>{e.label}</span>
          {idx < entries.length - 1 ? " · " : ""}
        </span>
      ))}
    </div>
  );
}

function ChatInstance({
  instance,
  knownConversations,
  onRemove,
  onChangeConversation,
  onChangeDraft,
  onNewConversation,
  onSend,
}: {
  instance: Instance;
  knownConversations: string[];
  onRemove: () => void;
  onChangeConversation: (id: string) => void;
  onChangeDraft: (v: string) => void;
  onNewConversation: () => void;
  onSend: () => void;
}) {
  const endRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [instance.messages.length, instance.sending]);

  return (
    <div className="flex h-full w-[420px] shrink-0 flex-col gap-4">
      <Card className="flex flex-col shadow-sm">
        <CardHeader>
          <div className="flex items-center justify-between gap-3">
            <CardTitle>Conversation</CardTitle>
            <Button variant="outline" size="icon" onClick={onRemove} aria-label="remove instance">
              <span className="text-sm font-semibold">–</span>
            </Button>
          </div>
        </CardHeader>
        <CardContent className="p-3 pt-0">
          <ConversationIdField
            value={instance.conversationId}
            onChange={onChangeConversation}
            options={knownConversations}
            onNew={onNewConversation}
          />
        </CardContent>
      </Card>

      <Card className="flex min-h-0 flex-1 flex-col">
        <CardHeader>
          <CardTitle>Chat</CardTitle>
        </CardHeader>
        <CardContent className="flex min-h-0 flex-1 flex-col gap-3 p-3 pt-0">
          <div className="min-h-0 flex-1 overflow-hidden rounded-2xl border border-border bg-background">
            <ScrollArea className="h-full">
              <div className="space-y-3 p-4">
                {instance.messages.map((m, idx) => (
                  <div
                    key={`${idx}-${m.role}`}
                    className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}
                  >
                    <div
                      className={cn(
                        "max-w-[85%] rounded-2xl border border-border bg-card p-3 text-xs shadow-sm",
                      )}
                    >
                      <div
                        className={cn(
                          "text-xs font-semibold text-muted-foreground",
                          m.role === "user" ? "text-right" : "text-left",
                        )}
                      >
                        {m.role === "user" ? "You" : m.role === "assistant" ? "Agent" : "System"}
                      </div>
                      <div className="mt-1 whitespace-pre-wrap">{m.content}</div>
                    </div>
                  </div>
                ))}

                {instance.sending ? (
                  <div className="flex justify-start">
                    <div className="max-w-[85%] rounded-2xl border border-border bg-card p-3 text-xs shadow-sm">
                      <div className="text-xs font-semibold text-muted-foreground">Agent</div>
                      <div className="mt-1 italic text-muted-foreground">Typing…</div>
                      {instance.stage ? <StageInline stage={instance.stage} /> : null}
                    </div>
                  </div>
                ) : null}
                <div ref={endRef} />
              </div>
            </ScrollArea>
          </div>

          <div className="flex items-end gap-3">
            <Textarea
              value={instance.draft}
              onChange={(e) => onChangeDraft(e.target.value)}
              placeholder=""
              className="h-9 min-h-0 resize-none"
              disabled={instance.sending}
            />
            <Button onClick={onSend} disabled={instance.sending || !instance.draft.trim()}>
              <Play className="h-4 w-4" />
              Send
            </Button>
          </div>

          {instance.error ? (
            <div className="text-sm text-destructive">{instance.error}</div>
          ) : null}
        </CardContent>
      </Card>

      <Card className="flex flex-col shadow-sm">
        <CardHeader>
          <CardTitle>Summary</CardTitle>
        </CardHeader>
        <CardContent className="p-3 pt-0">
          <div className="h-[88px] overflow-hidden rounded-xl border border-input bg-background">
            <ScrollArea className="h-full">
              <div className="px-3 py-2 text-xs text-muted-foreground whitespace-pre-wrap">
                {instance.summaryStatus === "waiting" && !instance.summary
                  ? "Wait Summarizing"
                  : instance.summary || ""}
              </div>
            </ScrollArea>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
