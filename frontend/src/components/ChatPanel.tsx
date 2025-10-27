import { useCallback, useMemo, useRef, useState, useEffect } from "react";
import { chat as chatApi, getData } from "../lib/endpoints";
import type { ChatMessage, ChatResponse } from "../types";

type Props = {
  onHighlightRows: (rows: number[]) => void;
  onReloadSheet: () => void;
  onAddPlot: (plot: { figure: any; title?: string }) => void;
};

export default function ChatPanel({ onHighlightRows, onReloadSheet, onAddPlot }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Hi! I understand your sheet. Try: “add a column for profit = revenue - cost”, “highlight rows where profit < 0”, or “pie chart of revenue by region”.",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<"auto" | "llm" | "rules">("auto");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const endRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const scrollToEnd = useCallback(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToEnd();
  }, [messages, scrollToEnd]);

  // Seed initial suggestions from current dataset if backend has not provided any yet
  useEffect(() => {
    if (suggestions.length > 0) return;
    (async () => {
      try {
        const data = await getData();
        const cols: string[] = (data.columns || []).map((c: any) => String(c.field));
        const lower = cols.map((c) => c.toLowerCase());
        const prefY = ["revenue", "sales", "amount", "profit", "q1", "q2"];
        const prefX = ["region", "product", "category", "name"];
        const y = prefY.find((n) => lower.includes(n)) || cols[1] || cols[0];
        const x = prefX.find((n) => lower.includes(n)) || cols[0];
        const s: string[] = [];
        if (y) s.push(`summarize total ${y}`);
        if (x && y) s.push(`pie chart of ${y} by ${x}`);
        if (s.length > 0) setSuggestions(s.slice(0, 2));
      } catch {}
    })();
  }, [suggestions.length]);

  const lastAssistant = useMemo(
    () => [...messages].reverse().find((m) => m.role === "assistant"),
    [messages]
  );

  const handleActions = useCallback(
    (resp: ChatResponse) => {
      if (!resp.actions) return;
      for (const a of resp.actions) {
        switch (a.type) {
          case "highlight_rows":
            if (Array.isArray(a.rows)) {
              onHighlightRows(a.rows);
            }
            break;
          case "add_column":
            // Backend already applied it; ask grid to reload
            onReloadSheet();
            break;
          case "plot":
            // Forward plot to App; App will switch to Diagrams tab and render it
            if (a.figure) {
              const title =
                (typeof a.figure?.layout?.title === "string"
                  ? a.figure?.layout?.title
                  : a.figure?.layout?.title?.text) ||
                (a.kind && a.x && a.y ? `${a.kind} of ${a.y} by ${a.x}` : "Chart");
              onAddPlot({ figure: a.figure, title });
            }
            break;
          case "update_cell":
            // Backend already applied the change; refresh grid to reflect
            onReloadSheet();
            break;
          case "add_row":
            // A new row was appended (optionally with values); refresh grid
            onReloadSheet();
            break;
          case "add_empty_column":
            // A new empty column was created; refresh grid
            onReloadSheet();
            break;
          case "delete_rows":
            // Rows were deleted; refresh grid
            onReloadSheet();
            break;
          case "delete_column":
            // A column was deleted; refresh grid
            onReloadSheet();
            break;
          default:
            break;
        }
      }
    },
    [onHighlightRows, onReloadSheet, onAddPlot]
  );

  const send = useCallback(async (override?: string) => {
    const text = (override ?? input).trim();
    if (!text || busy) return;
    setBusy(true);
    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");

    try {
      const resp = await chatApi({ message: text, mode });
      setMessages((m) => [
        ...m,
        { role: "assistant", content: resp.content, actions: resp.actions ?? [] },
      ]);
      try {
        setSuggestions((resp.suggestions ?? []).slice(0, 2));
      } catch {}
      handleActions(resp);
    } catch (e: any) {
      const errText = e?.message ? String(e.message) : "Request failed";
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Error: ${errText}` },
      ]);
    } finally {
      setBusy(false);
      // return focus to the input for faster follow-ups
      try { inputRef.current?.focus(); } catch {}
    }
  }, [busy, handleActions, input]);

  const quickPrompts = suggestions;

  return (
    <div className="h-full w-full flex flex-col min-h-0">
      <div className="flex-1 overflow-y-auto p-3 space-y-3 custom-scrollbar" role="log" aria-label="Chat messages">
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
            <div className={m.role === "user" ? "message-user" : "message-assistant"}>
              {m.content}
            </div>
          </div>
        ))}
        <div ref={endRef} />
        {/* Screen-reader live region for assistant messages */}
        <div className="sr-only" aria-live="polite" aria-atomic="true">
          {lastAssistant?.content || ""}
        </div>
      </div>

      <div className="px-3 py-3 border-t border-white/10 flex gap-2">
        <input
          ref={inputRef}
          className="flex-1 input-dark"
          placeholder='Ask e.g. "add a column for profit = revenue - cost"'
          aria-label="Chat message input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
        />
        <button
          type="button"
          className="btn-primary"
          onClick={() => send()}
          aria-label="Send message"
          title="Send message"
          disabled={busy}
        >
          {busy ? "Sending..." : "Send"}
        </button>
      </div>

      <div className="px-3 pb-3 pt-2 flex items-center justify-between gap-3 flex-wrap">
        <div className="flex gap-2 flex-wrap">
          {quickPrompts.map((q) => (
            <button
              key={q}
              className="btn-ghost text-xs"
              onClick={() => send(q)}
              title="Use this prompt"
            >
              {q}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-white/70 font-medium">Mode:</span>
          <select
            className="input-dark text-xs h-7 px-2 py-1"
            value={mode}
            onChange={(e) => setMode(e.target.value as "auto" | "llm" | "rules")}
            aria-label="Chat mode"
            title="Chat mode"
          >
            <option value="auto">Auto</option>
            <option value="llm">LLM</option>
            <option value="rules">Rules</option>
          </select>
        </div>
      </div>
    </div>
  );
}
