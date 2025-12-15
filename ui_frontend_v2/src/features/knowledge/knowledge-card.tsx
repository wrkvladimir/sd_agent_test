import * as React from "react";
import { Database, Play, RefreshCw, Upload } from "lucide-react";

import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { Dropzone } from "@/components/dropzone";
import { InfoBlock } from "@/components/info-block";
import { cn } from "@/lib/utils";

type OpStatus = "idle" | "loading" | "success" | "error";

export function KnowledgeCard({ className }: { className?: string }) {
  const [file, setFile] = React.useState<File | null>(null);
  const [output, setOutput] = React.useState<string>("");
  const [status, setStatus] = React.useState<OpStatus>("idle");
  const [statusLabel, setStatusLabel] = React.useState<string>("Ready");
  const [open, setOpen] = React.useState(false);

  async function runIngest() {
    if (!file) return;
    setStatus("loading");
    setStatusLabel("Ingesting…");
    try {
      const text = await file.text();
      const res = await api<any>("/api/ingest", {
        method: "POST",
        body: JSON.stringify({
          source_type: "inline_html",
          source_id: file.name || "uploaded_html",
          html: text,
        }),
      });
      setOutput(JSON.stringify(res, null, 2));
      setStatus("success");
      setStatusLabel("Success");
    } catch (e: any) {
      setStatus("error");
      setStatusLabel("Error");
      setOutput(e?.message ? String(e.message) : "Request failed");
    }
  }

  async function clearKnowledge() {
    setStatus("loading");
    setStatusLabel("Clearing…");
    try {
      const res = await api<any>("/api/clear", { method: "POST", body: "{}" });
      setOutput(JSON.stringify(res, null, 2));
      setStatus("success");
      setStatusLabel("Success");
    } catch (e: any) {
      setStatus("error");
      setStatusLabel("Error");
      setOutput(e?.message ? String(e.message) : "Request failed");
    }
  }

  async function getCurrentData() {
    setStatus("loading");
    setStatusLabel("Loading…");
    try {
      const res = await api<any>("/api/data");
      setOutput(JSON.stringify(res, null, 2));
      setStatus("success");
      setStatusLabel("Success");
    } catch (e: any) {
      setStatus("error");
      setStatusLabel("Error");
      setOutput(e?.message ? String(e.message) : "Request failed");
    }
  }

  const busy = status === "loading";

  return (
    <Card className={cn("flex min-h-0 flex-1 flex-col", className)}>
      <CardHeader>
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle>Knowledge Management</CardTitle>
          </div>
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
              <Button variant="outline">
                <Upload className="h-4 w-4" />
                Choose File
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Upload HTML</DialogTitle>
                <DialogDescription>Only `.html` is supported.</DialogDescription>
              </DialogHeader>

              <Dropzone
                accept=".html,text/html"
                title="Drag & drop an .html file"
                description="Or choose a file from your computer."
                onFile={(f) => {
                  setFile(f);
                  setOpen(false);
                }}
              />

              <InfoBlock title="Expected HTML Structure (required)">
                <div className="space-y-2">
                  <div>
                    Container: element with `aria-label="База знаний (чанки)"`.
                  </div>
                  <div>
                    Articles: `article.kb-item` with optional attributes `data-id` and `data-date`.
                  </div>
                  <div>
                    Content: `h2` (title) + multiple `p` (paragraphs). Each paragraph is chunked and embedded.
                  </div>
                  <div>
                    Extracted fields: title, text, source_id, source_date, source_label, source_document_id.
                  </div>
                  <div>Ignored: content outside the container and empty paragraphs.</div>
                </div>
              </InfoBlock>
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>

      <CardContent className="flex min-h-0 flex-1 flex-col gap-2 p-3 pt-0">
        <div className="grid grid-cols-3 gap-2">
          <Button
            size="sm"
            onClick={runIngest}
            disabled={!file || busy}
            className="h-auto whitespace-normal py-2 leading-snug"
          >
            <Play className="h-4 w-4" />
            Run Ingest
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={clearKnowledge}
            disabled={busy}
            className="h-auto whitespace-normal py-2 leading-snug"
          >
            <RefreshCw className={cn("h-4 w-4", busy && "animate-spin")} />
            Clear Knowledge
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={getCurrentData}
            disabled={busy}
            className="h-auto whitespace-normal py-2 leading-snug"
          >
            <Database className="h-4 w-4" />
            Get Current Data
          </Button>
        </div>

        <div className="flex items-center justify-between">
          <div className="text-sm font-medium">Result</div>
          <div
            className={cn(
              "text-sm",
              status === "success" && "text-foreground",
              status === "error" && "text-destructive",
              status === "loading" && "text-muted-foreground",
            )}
          >
            {statusLabel}
          </div>
        </div>

        <Textarea
          readOnly
          value={output}
          placeholder="Operation output will appear here…"
          className="min-h-0 flex-1 resize-none font-mono text-[11px]"
        />
      </CardContent>
    </Card>
  );
}
