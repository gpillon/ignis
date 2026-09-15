import { useCallback, useEffect, useState } from "react";
import { forgetKey, useAuth } from "../api/auth.ts";
import { useModel } from "../api/model.ts";
import { Composer } from "../conversation/Composer.tsx";
import { type OpenAgent, Transcript } from "../conversation/Transcript.tsx";
import { contextUsage } from "../metrics/context.ts";
import { Readout } from "../metrics/Readout.tsx";
import { SessionLog } from "../sessions/SessionLog.tsx";
import { SessionsPanel } from "../sessions/SessionsPanel.tsx";
import { DEFAULT_SETTINGS, type PlaygroundSettings } from "../settings/defaults.ts";
import { SettingsPanel } from "../settings/SettingsPanel.tsx";
import { AgentReader } from "../tools/agents/AgentReader.tsx";
import { MemoryManager } from "../tools/local/MemoryManager.tsx";
import { AGENT_SYSTEM_PROMPT, type AgentRun } from "../tools/agents/agents.ts";
import { ALL_TOOLS, type ToolsState } from "../tools/index.ts";
import { Header } from "./Header.tsx";
import { KeyPage } from "./KeyPage.tsx";
import { useConversation } from "./useConversation.ts";

// The Playground (GitHub #164): a streaming chat against ignis's own
// /v1/chat/completions, with per-request figures measured in the browser.
// Sessions, settings and figures live in memory; a reload starts over.
// This is the page's layout; the conversation loop is useConversation.

type Drawer = "sessions" | "settings" | null;

export function App() {
  const auth = useAuth();
  const model = useModel();
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [tools, setTools] = useState<ToolsState>(ALL_TOOLS);
  const [markdown, setMarkdown] = useState(true);
  const [input, setInput] = useState("");
  const [logOpen, setLogOpen] = useState(false);
  const [reader, setReader] = useState<OpenAgent | null>(null);
  const closeReader = useCallback(() => setReader(null), []);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const closeMemory = useCallback(() => setMemoryOpen(false), []);
  const [drawer, setDrawer] = useState<Drawer>(null);
  const chat = useConversation({ model, settings, tools });
  const { active } = chat;

  const readerRun: AgentRun | undefined = reader
    ? active.messages.find((m) => m.id === reader.messageId)?.agents?.find((r) => r.callId === reader.callId)
    : undefined;

  useEffect(() => {
    if (!drawer) return;
    const close = (e: globalThis.KeyboardEvent) => e.key === "Escape" && setDrawer(null);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [drawer]);

  function selectSession(id: number) {
    chat.selectSession(id);
    setDrawer(null);
    setReader(null);
  }

  function newSession() {
    chat.newSession();
    setDrawer(null);
  }

  function send() {
    const text = input.trim();
    if (!text || !chat.canRun) return;
    setInput("");
    chat.send(text);
  }

  const set = <K extends keyof PlaygroundSettings>(key: K, value: PlaygroundSettings[K]) =>
    setSettings((s) => ({ ...s, [key]: value }));

  // A 401 anywhere shows the key prompt in place of the page; App stays
  // mounted, so the sessions are still here once the key is in.
  if (auth.needsKey) return <KeyPage rejected={auth.rejected} />;

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <Header model={model} busy={chat.busy} onOpen={setDrawer} onForgetKey={auth.key ? forgetKey : undefined} onOpenMemory={() => setMemoryOpen(true)} />

      <div className="relative flex min-h-0 flex-1">
        {drawer && (
          <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => setDrawer(null)} aria-hidden />
        )}

        <SessionsPanel
          open={drawer === "sessions"}
          list={chat.list}
          activeId={active.id}
          streamingId={chat.streamingId}
          onNew={newSession}
          onSelect={selectSession}
          onRemove={chat.deleteSession}
        />

        <main className="flex min-w-0 flex-1 flex-col">
          <Transcript
            session={active}
            model={model}
            markdown={markdown}
            canRerun={chat.canRun}
            following={chat.following}
            openAgent={reader}
            onOpenAgent={setReader}
            onRerun={chat.rerun}
            onSave={chat.saveEdit}
            onFork={chat.fork}
            onAnswer={chat.answer}
          />
          <Composer
            value={input}
            onChange={setInput}
            ready={model.state === "ready"}
            busy={chat.busy}
            streamingHere={chat.streamingId === active.id}
            usage={contextUsage(active.log, settings.maxTokens, model.state === "ready" ? model.contextLimit : null)}
            onSend={send}
            onStop={chat.stop}
            canAttach={tools.enabled && tools.readFiles}
            attachments={active.attachments}
            attachError={chat.attachError}
            onAttach={(files) => void chat.attach(files)}
            onDetach={chat.detach}
          />
        </main>

        <SettingsPanel
          open={drawer === "settings"}
          settings={settings}
          set={set}
          tools={tools}
          onToolsChange={setTools}
          markdown={markdown}
          onMarkdownChange={setMarkdown}
          attachments={active.attachments}
          onOpenMemory={() => setMemoryOpen(true)}
        />
      </div>

      <SessionLog rows={active.log} open={logOpen} onToggle={() => setLogOpen((o) => !o)} />

      {memoryOpen && <MemoryManager onClose={closeMemory} />}

      {readerRun && (
        <>
          <div className="fixed inset-0 z-40 bg-[#1c2026]/40" onClick={closeReader} aria-hidden />
          <AgentReader
            key={readerRun.callId}
            run={readerRun}
            markdown={markdown}
            systemPrompt={readerRun.systemPrompt ?? AGENT_SYSTEM_PROMPT}
            figures={readerRun.figures && <Readout figures={readerRun.figures} />}
            onClose={closeReader}
          />
        </>
      )}
    </div>
  );
}
