import { useCallback, useEffect, useState } from "react";
import { forgetKey, useAuth } from "../api/auth.ts";
import { useModel } from "../api/model.ts";
import { Composer } from "../conversation/Composer.tsx";
import { imageFromFile, type PromptImage } from "../conversation/images.ts";
import { DecideView, type Drawer as DecideDrawer } from "../decide/DecideView.tsx";
import { type OpenAgent, Transcript } from "../conversation/Transcript.tsx";
import { contextUsage } from "../metrics/context.ts";
import { Readout } from "../metrics/Readout.tsx";
import { MonitorView } from "../monitor/MonitorView.tsx";
import { useMonitor, useMonitorVisible } from "../monitor/useMonitor.ts";
import { SessionLog } from "../sessions/SessionLog.tsx";
import { SessionsPanel } from "../sessions/SessionsPanel.tsx";
import { DEFAULT_SETTINGS, type PlaygroundSettings } from "../settings/defaults.ts";
import { SettingsPanel } from "../settings/SettingsPanel.tsx";
import { AgentReader } from "../tools/agents/AgentReader.tsx";
import { MemoryManager } from "../tools/local/MemoryManager.tsx";
import { AGENT_SYSTEM_PROMPT, type AgentRun } from "../tools/agents/agents.ts";
import { ALL_TOOLS, type ToolsState } from "../tools/index.ts";
import { Header, type View } from "./Header.tsx";
import { KeyPage } from "./KeyPage.tsx";
import { STORED_FLAGS, useStoredFlag } from "./storedFlag.ts";
import { useConversation } from "./useConversation.ts";

// The Playground (GitHub #164): a streaming chat against ignis's own
// /v1/chat/completions, with per-request figures measured in the browser.
// Sessions, settings and figures live in memory; a reload starts over, apart
// from the switches this browser stores (`storedFlag.ts`).
// This is the page's layout; the conversation loop is useConversation.
// The header switches between the chat, the Decide tab (GitHub #247) and —
// when ignis serves metrics — the Monitor (GitHub #165). Whichever is showing,
// the chat stays mounted underneath, so a streaming reply carries on.

type Drawer = "sessions" | "settings" | null;

export function App() {
  const auth = useAuth();
  const model = useModel();
  const monitorVisible = useMonitorVisible();
  const [view, setView] = useState<View>("chat");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [tools, setTools] = useState<ToolsState>(ALL_TOOLS);
  const [markdown, setMarkdown] = useState(true);
  // Off: the page runs one turn at a time, whichever session it is in. Stored,
  // so turning it on survives a reload rather than looking like it never took.
  const [parallel, setParallel] = useStoredFlag(STORED_FLAGS.parallel, false);
  const [input, setInput] = useState("");
  // The images the next prompt will carry. They belong to the box, not to a
  // session: once sent they live on the user message, and the box is empty again.
  const [images, setImages] = useState<PromptImage[]>([]);
  const [imageError, setImageError] = useState<string | null>(null);
  const [logOpen, setLogOpen] = useState(false);
  const [reader, setReader] = useState<OpenAgent | null>(null);
  const closeReader = useCallback(() => setReader(null), []);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const closeMemory = useCallback(() => setMemoryOpen(false), []);
  const [drawer, setDrawer] = useState<Drawer>(null);
  // The bench has its own two drawers below `lg`; the header's buttons open
  // whichever view is showing.
  const [decideDrawer, setDecideDrawer] = useState<DecideDrawer>(null);
  const chat = useConversation({ model, settings, tools, parallel });
  const { active } = chat;
  const monitoring = view === "monitor" && monitorVisible;
  const deciding = view === "decide";

  const readerRun: AgentRun | undefined = reader
    ? active.messages.find((m) => m.id === reader.messageId)?.agents?.find((r) => r.callId === reader.callId)
    : undefined;

  // Metrics going away drops back to the chat *from the Monitor*, so their
  // return never flips the page on its own — and a reader in the Decide tab,
  // which needs no metrics, is left where they are.
  useEffect(() => {
    if (!monitorVisible) setView((current) => (current === "monitor" ? "chat" : current));
  }, [monitorVisible]);

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
    if ((!text && images.length === 0) || !chat.canRun) return;
    setInput("");
    setImages([]);
    setImageError(null);
    chat.send(text, images);
  }

  /** Picked or pasted files as prompt images; the ones that cannot be read are named instead. */
  async function addImages(files: File[]) {
    const results = await Promise.all(files.map(imageFromFile));
    const errors = results.flatMap((r) => (r.ok ? [] : [r.error]));
    setImageError(errors.length > 0 ? errors.join(" ") : null);
    const added = results.flatMap((r) => (r.ok ? [r.image] : []));
    if (added.length > 0) setImages((current) => [...current, ...added]);
  }

  const set = <K extends keyof PlaygroundSettings>(key: K, value: PlaygroundSettings[K]) =>
    setSettings((s) => ({ ...s, [key]: value }));

  // A 401 anywhere shows the key prompt in place of the page; App stays
  // mounted, so the sessions are still here once the key is in.
  if (auth.needsKey) return <KeyPage rejected={auth.rejected} />;

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <Header model={model} busy={chat.busy} onOpen={(which) => (deciding ? setDecideDrawer(which) : setDrawer(which))} onForgetKey={auth.key ? forgetKey : undefined} onOpenMemory={() => setMemoryOpen(true)} view={monitoring ? "monitor" : deciding ? "decide" : "chat"} onView={setView} monitorAvailable={monitorVisible} />

      {monitoring && <Monitor />}

      {/* Mounted whichever view is showing, like the chat below it: the tab
          holds a whole typed request, and a glance at the Monitor is not a
          reason to lose it. Nothing here fetches on mount. */}
      <div className={deciding ? "contents" : "hidden"}>
        <DecideView ready={model.state === "ready"} drawer={decideDrawer} onDrawer={setDecideDrawer} />
      </div>

      <div className={monitoring || deciding ? "hidden" : "contents"}>
        <div className="relative flex min-h-0 flex-1">
          {drawer && (
            <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => setDrawer(null)} aria-hidden />
          )}

          <SessionsPanel
            open={drawer === "sessions"}
            list={chat.list}
            activeId={active.id}
            streaming={chat.streaming}
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
              busy={chat.sendBlocked}
              streamingHere={chat.streamingHere}
              usage={contextUsage(active.log, settings.maxTokens, model.state === "ready" ? model.contextLimit : null)}
              onSend={send}
              onStop={chat.stop}
              canAttach={tools.enabled && tools.readFiles}
              attachments={active.attachments}
              attachError={chat.attachError}
              onAttach={(files) => void chat.attach(files)}
              onDetach={chat.detach}
              images={images}
              imageError={imageError}
              onAddImages={(files) => void addImages(files)}
              onRemoveImage={(index) => setImages((current) => current.filter((_, i) => i !== index))}
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
            parallel={parallel}
            onParallelChange={setParallel}
            attachments={active.attachments}
            onOpenMemory={() => setMemoryOpen(true)}
          />
        </div>

        <SessionLog rows={active.log} open={logOpen} onToggle={() => setLogOpen((o) => !o)} />
      </div>

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

/** The Monitor subscribes on its own, so a scrape re-renders it and not the chat. */
function Monitor() {
  return <MonitorView state={useMonitor()} />;
}
