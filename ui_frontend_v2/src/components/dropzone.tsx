import * as React from "react";
import { Upload } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export function Dropzone({
  accept,
  onFile,
  title,
  description,
  className,
}: {
  accept: string;
  onFile: (file: File) => void;
  title: string;
  description?: string;
  className?: string;
}) {
  const [dragOver, setDragOver] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement | null>(null);

  return (
    <div
      className={cn(
        "rounded-2xl border border-border bg-background p-4 shadow-sm",
        dragOver && "bg-accent",
        className,
      )}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) onFile(file);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="sr-only"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          if (inputRef.current) inputRef.current.value = "";
        }}
      />
      <div className="flex items-start gap-3">
        <div className="mt-0.5 rounded-xl border border-border bg-card p-2">
          <Upload className="h-4 w-4" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium">{title}</div>
          {description ? (
            <div className="mt-1 text-sm text-muted-foreground">{description}</div>
          ) : null}
          <div className="mt-3 flex min-w-0 items-center gap-3">
            <Button type="button" variant="outline" onClick={() => inputRef.current?.click()}>
              Choose file
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
