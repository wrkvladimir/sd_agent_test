import * as React from "react";
import { ChevronDown, Plus } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

export function ConversationIdField({
  value,
  onChange,
  options,
  onNew,
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  options: string[];
  onNew: () => void;
  className?: string;
}) {
  const [open, setOpen] = React.useState(false);

  return (
    <div className={cn("relative", className)}>
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="conversation_id"
        className="pr-10"
      />
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="absolute right-1 top-1 h-7 w-7 rounded-lg"
            aria-label="open conversation list"
          >
            <ChevronDown className="h-4 w-4 opacity-70" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[360px] p-2" align="end">
          <Command>
            <CommandInput placeholder="Search conversation_id…" />
            <CommandList>
              <CommandEmpty>No results.</CommandEmpty>
              <CommandGroup>
                <CommandItem
                  onSelect={() => {
                    setOpen(false);
                    onNew();
                  }}
                >
                  <Plus className="mr-2 h-4 w-4" />
                  New conversation_id
                </CommandItem>
              </CommandGroup>
              <CommandGroup>
                {options.map((id) => (
                  <CommandItem
                    key={id}
                    onSelect={() => {
                      onChange(id);
                      setOpen(false);
                    }}
                  >
                    {id}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}

