"use client";

import { useEffect, useRef, useState } from "react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Sparkles,
  Send,
  Bot,
  User,
  FileText,
  Copy,
  Check,
  Loader2,
  RefreshCw,
  Trash2,
  ExternalLink,
  Landmark,
  Settings,
} from "lucide-react";

// Backend base URL precedence:
//   1. A URL the user set at runtime (stored in localStorage) — lets you point
//      the UI at a live model without rebuilding/redeploying.
//   2. NEXT_PUBLIC_API_URL (build-time env).
//   3. Same-origin /api/* (Next.js rewrites to the backend).
const ENV_API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";
const API_URL_KEY = "finlens.modelUrl";

function getApiUrl(): string {
  try {
    const saved = window.localStorage.getItem(API_URL_KEY);
    if (saved && saved.trim()) return saved.trim();
  } catch {}
  return ENV_API_URL;
}
function setApiUrl(url: string) {
  try {
    window.localStorage.setItem(API_URL_KEY, url.trim());
  } catch {}
}

type Source = {
  source_doc: string;
  url: string;
  snippet: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const SUGGESTIONS = [
  "What is CKYC and how is it different from regular KYC?",
  "What are the AML reporting obligations under PMLA 2002?",
  "What is the difference between VKYC and e-KYC?",
  "What is the process for CERSAI registration?",
];

function StreamIcon({ active }: { active: boolean }) {
  return active ? (
    <Loader2 className="h-4 w-4 animate-spin" />
  ) : (
    <Sparkles className="h-4 w-4" />
  );
}

function SourcesList({ sources }: { sources: Source[] }) {
  const [copied, setCopied] = useState<string | null>(null);
  return (
    <div className="mt-3 rounded-xl border bg-muted/40 p-3">
      <div className="mb-2 flex items-center gap-2">
        <FileText className="h-3.5 w-3.5 text-muted-foreground" />
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Sources
        </p>
      </div>
      <ul className="space-y-1.5">
        {sources.map((s, i) => (
          <li key={i}>
            <a
              href={s.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-start gap-2 rounded-md px-1 py-1 text-sm text-foreground/90 transition-colors hover:bg-accent hover:text-accent-foreground"
            >
              <Badge variant="secondary" className="mt-0.5 shrink-0">
                {i + 1}
              </Badge>
              <span className="min-w-0 flex-1">
                <span className="line-clamp-1 font-medium">
                  {s.source_doc}
                </span>
                <span className="line-clamp-1 text-xs text-muted-foreground">
                  {s.snippet}
                </span>
              </span>
              <ExternalLink className="mt-1 h-3 w-3 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-7 w-7 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      aria-label="Copy response"
    >
      {copied ? (
        <Check className="h-3.5 w-3.5" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
    </Button>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [modelUrl, setModelUrl] = useState<string>(() => getApiUrl());
  const [modelUrlSaved, setModelUrlSaved] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isThinking]);

  async function send(message?: string) {
    const text = (message ?? input).trim();
    if (!text || isStreaming) return;

    setError(null);
    setInput("");
    const userMsg: Message = { role: "user", content: text };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setMessages([...updated, { role: "assistant", content: "" }]);
    setIsStreaming(true);
    setIsThinking(true);

    const history = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${getApiUrl()}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        throw new Error(`Server error: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finalSources: Source[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const evt of events) {
          const line = evt.trim();
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6);
          if (payload === "[DONE]") continue;

          let data;
          try {
            data = JSON.parse(payload);
          } catch {
            continue;
          }

          if (data.type === "token") {
            setIsThinking(false);
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              next[next.length - 1] = {
                ...last,
                content: last.content + data.content,
              };
              return next;
            });
          } else if (data.type === "sources") {
            finalSources = data.sources ?? [];
          }
        }
      }

      if (finalSources.length) {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          next[next.length - 1] = { ...last, sources: finalSources };
          return next;
        });
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.content === "") {
            next.pop();
          }
          return next;
        });
      } else {
        setError((err as Error).message);
        setMessages((prev) => prev.slice(0, -1));
      }
    } finally {
      setIsStreaming(false);
      setIsThinking(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function reset() {
    stop();
    setMessages([]);
    setError(null);
    setIsThinking(false);
    setIsStreaming(false);
  }

  const hasMessages = messages.length > 0;

  return (
    <TooltipProvider>
      <div className="flex h-full flex-1 flex-col bg-zinc-50 dark:bg-zinc-950">
        <header className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur">
          <div className="mx-auto flex h-14 w-full max-w-3xl items-center justify-between px-4">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Landmark className="h-4 w-4" />
              </div>
              <div>
                <h1 className="text-sm font-semibold leading-none">
                  FinLens
                </h1>
                <p className="text-xs text-muted-foreground">
                  Fintech compliance copilot
                </p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={reset}
                      disabled={!hasMessages && !isStreaming}
                      aria-label="New chat"
                    />
                  }
                >
                  <RefreshCw className="h-4 w-4" />
                </TooltipTrigger>
                <TooltipContent>New chat</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setMessages([])}
                      disabled={!hasMessages}
                      aria-label="Clear messages"
                    />
                  }
                >
                  <Trash2 className="h-4 w-4" />
                </TooltipTrigger>
                <TooltipContent>Clear chat</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        setModelUrl(getApiUrl());
                        setModelUrlSaved(false);
                        setShowSettings((v) => !v);
                      }}
                      aria-label="Settings"
                    />
                  }
                >
                  <Settings className="h-4 w-4" />
                </TooltipTrigger>
                <TooltipContent>Model URL</TooltipContent>
              </Tooltip>
            </div>
          </div>
          {showSettings && (
            <div className="border-b bg-background/95 backdrop-blur">
              <div className="mx-auto w-full max-w-3xl px-4 py-3">
                <div className="flex flex-col gap-2 rounded-xl border bg-muted/40 p-3 sm:flex-row sm:items-end">
                  <div className="flex-1 space-y-1">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Model backend URL
                    </label>
                    <input
                      value={modelUrl}
                      onChange={(e) => {
                        setModelUrl(e.target.value);
                        setModelUrlSaved(false);
                      }}
                      placeholder="https://xxx.trycloudflare.com"
                      className="w-full rounded-md border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring"
                    />
                    <p className="text-xs text-muted-foreground">
                      Paste the live Kaggle tunnel URL (no trailing slash). Used
                      for /api/chat. Stored locally in your browser.
                    </p>
                  </div>
                  <Button
                    size="sm"
                    onClick={() => {
                      setApiUrl(modelUrl);
                      setModelUrlSaved(true);
                      setTimeout(() => setModelUrlSaved(false), 1500);
                    }}
                  >
                    <Check className="h-4 w-4" />
                    {modelUrlSaved ? "Saved" : "Save"}
                  </Button>
                </div>
              </div>
            </div>
          )}
        </header>

        <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4 py-6">
          {!hasMessages ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-8 py-12">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-lg">
                <Landmark className="h-7 w-7" />
              </div>
              <div className="text-center">
                <h2 className="text-2xl font-semibold tracking-tight">
                  Ask about Indian fintech regulation
                </h2>
                <p className="mt-2 max-w-md text-sm text-muted-foreground">
                  Grounded in RBI master directions, FIU-IND obligations,
                  SEBI reports, and PMLA 2002 — with citations.
                </p>
              </div>
              <div className="grid w-full max-w-lg grid-cols-1 gap-2 sm:grid-cols-2">
                {SUGGESTIONS.map((s) => (
                  <Button
                    key={s}
                    variant="outline"
                    className="h-auto justify-start whitespace-normal px-3 py-2.5 text-left text-sm"
                    onClick={() => send(s)}
                    disabled={isStreaming}
                  >
                    <Sparkles className="mr-2 h-4 w-4 shrink-0 text-primary" />
                    {s}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex-1 space-y-6">
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={
                    m.role === "user"
                      ? "flex justify-end"
                      : "flex justify-start"
                  }
                >
                  <div
                    className={
                      m.role === "user"
                        ? "group flex max-w-[85%] items-start gap-3 sm:max-w-[75%]"
                        : "group flex max-w-full items-start gap-3"
                    }
                  >
                    {m.role === "assistant" && (
                      <Avatar className="mt-0.5 h-8 w-8">
                        <AvatarFallback className="bg-primary text-primary-foreground">
                          <Bot className="h-4 w-4" />
                        </AvatarFallback>
                      </Avatar>
                    )}
                    <div
                      className={
                        m.role === "user"
                          ? "rounded-2xl rounded-tr-sm bg-primary px-4 py-3 text-sm text-primary-foreground"
                          : "min-w-0 flex-1"
                      }
                    >
                      {m.role === "user" ? (
                        <p className="whitespace-pre-wrap leading-relaxed">
                          {m.content}
                        </p>
                      ) : (
                        <>
                          <div className="flex items-start gap-2">
                            <div className="flex-1 rounded-2xl rounded-tl-sm border bg-background px-4 py-3 text-sm leading-relaxed">
                              {m.content ? (
                                <p className="whitespace-pre-wrap">
                                  {m.content}
                                </p>
                              ) : isThinking ? (
                                <ThinkingIndicator />
                              ) : null}
                              {m.content && m.sources && (
                                <SourcesList sources={m.sources} />
                              )}
                            </div>
                            {m.content && (
                              <CopyButton text={m.content} />
                            )}
                          </div>
                        </>
                      )}
                    </div>
                    {m.role === "user" && (
                      <Avatar className="mt-0.5 h-8 w-8">
                        <AvatarFallback className="bg-zinc-200 text-zinc-800 dark:bg-zinc-800 dark:text-zinc-100">
                          <User className="h-4 w-4" />
                        </AvatarFallback>
                      </Avatar>
                    )}
                  </div>
                </div>
              ))}
              <div ref={scrollRef} />
            </div>
          )}

          {error && (
            <div className="mt-4 rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {error} — is the server running? Try{" "}
              <code className="rounded bg-muted px-1 py-0.5">
                python server.py
              </code>
            </div>
          )}
        </main>

        <footer className="sticky bottom-0 border-t bg-background/80 backdrop-blur">
          <div className="mx-auto w-full max-w-3xl px-4 py-4">
            <Card>
              <CardContent className="pt-4">
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      send();
                    }
                  }}
                  placeholder="Ask about CKYC, AML, VKYC, CERSAI, PMLA..."
                  rows={1}
                  className="max-h-40 min-h-[44px] resize-none border-0 bg-transparent p-0 shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
                />
              </CardContent>
              <CardFooter className="justify-between border-t px-4 py-3">
                <p className="text-xs text-muted-foreground">
                  Grounded in real regulatory docs · answers cite sources
                </p>
                <div className="flex items-center gap-2">
                  {isStreaming && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={stop}
                      className="gap-2"
                    >
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      Stop
                    </Button>
                  )}
                  <Button
                    size="sm"
                    onClick={() => send()}
                    disabled={!input.trim() || isStreaming}
                    className="gap-2"
                  >
                    <StreamIcon active={isStreaming} />
                    Send
                  </Button>
                </div>
              </CardFooter>
            </Card>
          </div>
        </footer>
      </div>
    </TooltipProvider>
  );
}

function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-1.5 py-1">
      {[0, 150, 300].map((d) => (
        <span
          key={d}
          className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/60"
          style={{ animationDelay: `${d}ms` }}
        />
      ))}
    </div>
  );
}
