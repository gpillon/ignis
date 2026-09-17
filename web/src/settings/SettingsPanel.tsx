import { useState } from "react";
import { REASONING_EFFORTS } from "../api/request.ts";
import { ignisPrompt, setAllTools, type ToolsState, toolsInUse } from "../tools/index.ts";
import type { Attachment } from "../tools/local/attachments.ts";
import { useMemoryNotes } from "../tools/local/memory.ts";
import { setTavilyKey, useTavilyKey } from "../tools/web/tavilyKey.ts";
import { caption, field } from "../ui/classes.ts";
import { Segmented } from "../ui/Segmented.tsx";
import { Slider } from "../ui/Slider.tsx";
import { Switch } from "../ui/Switch.tsx";
import { EFFORT_LABELS, type PlaygroundSettings } from "./defaults.ts";
import { SystemPromptField } from "./SystemPromptField.tsx";

type Tab = "general" | "tools";

/** The right-hand panel, in two tabs: request settings and display, and the tools. A drawer below `lg`, shown when `open`. */
export function SettingsPanel(props: {
  open: boolean;
  settings: PlaygroundSettings;
  set: <K extends keyof PlaygroundSettings>(key: K, value: PlaygroundSettings[K]) => void;
  tools: ToolsState;
  onToolsChange: (tools: ToolsState) => void;
  markdown: boolean;
  onMarkdownChange: (on: boolean) => void;
  /** The active session's files, which the ignis prompt lists. */
  attachments: Attachment[];
  onOpenMemory: () => void;
}) {
  const [tab, setTab] = useState<Tab>("general");
  const { tools } = props;
  const toolsOn = toolsInUse(tools);
  return (
    <aside
      aria-label="Settings"
      className={`fixed inset-y-0 right-0 z-40 flex w-72 flex-col overflow-y-auto border-l border-line bg-ground px-5 pb-5 transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:w-68 lg:translate-x-0 ${props.open ? "translate-x-0" : "max-lg:invisible max-lg:translate-x-full"}`}
    >
      <div role="tablist" aria-label="Settings sections" className="sticky top-0 z-10 -mx-5 mb-5 flex gap-5 border-b border-line bg-ground px-5 pt-4">
        {(["general", "tools"] as const).map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            id={`settings-tab-${t}`}
            aria-selected={tab === t}
            aria-controls="settings-tab-panel"
            onClick={() => setTab(t)}
            className={`-mb-px flex items-center gap-1.5 border-b-2 pb-2 font-display text-[13px] font-semibold ${tab === t ? "border-ember text-ink" : "border-transparent text-ash hover:text-ink"}`}
          >
            {t === "general" ? "General" : "Tools"}
            {t === "tools" && (
              <span className="font-display text-xs font-medium tabular-nums text-ash" aria-label={`${toolsOn} on`}>
                {toolsOn}
              </span>
            )}
          </button>
        ))}
      </div>
      <div id="settings-tab-panel" role="tabpanel" aria-labelledby={`settings-tab-${tab}`} className="flex flex-1 flex-col gap-6">
        {tab === "general" ? <GeneralSettings {...props} /> : <ToolsSetting tools={tools} onChange={props.onToolsChange} onOpenMemory={props.onOpenMemory} />}
      </div>
    </aside>
  );
}

function GeneralSettings(props: {
  settings: PlaygroundSettings;
  set: <K extends keyof PlaygroundSettings>(key: K, value: PlaygroundSettings[K]) => void;
  tools: ToolsState;
  markdown: boolean;
  onMarkdownChange: (on: boolean) => void;
  attachments: Attachment[];
}) {
  const { settings, set } = props;
  const notes = useMemoryNotes();
  // ignis refuses greedy sampling with a top_p it would ignore.
  const greedyConflict = settings.temperature === 0 && settings.topP !== 1;
  return (
    <>
      <SystemPromptField
        value={settings.systemPrompt}
        onChange={(v) => set("systemPrompt", v)}
        ignis={ignisPrompt(props.tools, { notes, attachments: props.attachments })}
      />

      <Segmented
        legend="Thinking"
        name="effort"
        value={settings.reasoningEffort}
        options={REASONING_EFFORTS.map((effort) => ({ value: effort, label: EFFORT_LABELS[effort] }))}
        onChange={(v) => set("reasoningEffort", v)}
      />

      <div className="flex flex-col gap-4">
        <Slider label="Temperature" name="temperature" min={0} max={2} step={0.05} value={settings.temperature} onChange={(v) => set("temperature", v)} />
        <Slider label="top_p" name="top-p" min={0} max={1} step={0.05} value={settings.topP} onChange={(v) => set("topP", v)} />
        {greedyConflict && (
          <p className="border-l-2 border-fault pl-2 text-xs leading-snug text-fault">
            Temperature 0 needs top_p at 1, or ignis answers 400.
          </p>
        )}
      </div>

      <label className="flex flex-col gap-2">
        <span className={caption}>Max tokens</span>
        <input
          className={`${field} font-display tabular-nums`}
          type="number"
          name="max-tokens"
          min={1}
          placeholder="Engine cap"
          value={settings.maxTokens ?? ""}
          onChange={(e) => set("maxTokens", e.target.value === "" ? null : Number(e.target.value))}
        />
      </label>

      <Segmented
        legend="Lane tag"
        name="lane"
        value={settings.laneTag}
        options={[
          { value: "interactive", label: "Interactive" },
          { value: "agent", label: "Agent" },
        ]}
        onChange={(v) => set("laneTag", v)}
      />

      <MarkdownSetting on={props.markdown} onChange={props.onMarkdownChange} />
    </>
  );
}

function ToolsSetting({
  tools,
  onChange,
  onOpenMemory,
}: {
  tools: ToolsState;
  onChange: (tools: ToolsState) => void;
  onOpenMemory: () => void;
}) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="sr-only">Tools</legend>
      <div className="mb-2 flex items-center justify-between gap-3">
        <span id="tools-all-label" className={caption}>
          All tools
        </span>
        <Switch on={tools.enabled} onChange={(on) => onChange(setAllTools(tools, on))} labelledBy="tools-all-label" />
      </div>
      {tools.enabled && (
        <div className="flex flex-col gap-4">
          <ToolRow
            id="agents"
            label="Agents"
            on={tools.agents}
            onChange={(agents) => onChange({ ...tools, agents })}
            description="The model can hand sub-tasks to agents that run in parallel on agent lanes. Agents get the other tools that are on, except asking you."
          />
          <ToolRow
            id="web"
            label="Web"
            on={tools.web}
            onChange={(web) => onChange({ ...tools, web })}
            description="The model can search the web and read pages, from this browser."
          />
          {tools.web && <TavilyKeyField />}
          <ToolRow
            id="ask-user"
            label="Ask me"
            on={tools.askUser}
            onChange={(askUser) => onChange({ ...tools, askUser })}
            description="The model can stop to ask you a question, with answers to pick or your own."
          />
          <ToolRow
            id="date-time"
            label="Date and time"
            on={tools.dateTime}
            onChange={(dateTime) => onChange({ ...tools, dateTime })}
            description="The day, date and time this session began, in this browser's time zone, go into the ignis system prompt."
          />
          {tools.dateTime && (
            <div className="-mt-2 flex flex-col gap-1.5 border-l-2 border-line pl-3">
              <div className="flex items-start justify-between gap-3">
                <div className="flex flex-col gap-0.5">
                  <span id="tool-date-live-label" className="font-display text-xs font-semibold text-ink">
                    Update every prompt
                  </span>
                  <span className="text-xs leading-snug text-ash">
                    Each turn carries the moment it was sent, as a developer message after the conversation, instead of the session's
                    moment in the system prompt.
                  </span>
                </div>
                <Switch
                  on={tools.dateTimeLive}
                  onChange={(dateTimeLive) => onChange({ ...tools, dateTimeLive })}
                  labelledBy="tool-date-live-label"
                />
              </div>
            </div>
          )}
          <ToolRow
            id="run-js"
            label="Run JavaScript"
            on={tools.runJs}
            onChange={(runJs) => onChange({ ...tools, runJs })}
            description="The model can run JavaScript in a sandboxed worker with no network, for exact results."
          />
          {tools.runJs && (
            <div className="-mt-2 flex flex-col gap-1.5 border-l-2 border-line pl-3">
              <div className="flex items-start justify-between gap-3">
                <div className="flex flex-col gap-0.5">
                  <span id="tool-js-check-label" className="font-display text-xs font-semibold text-ink">
                    Safety check
                  </span>
                  <span className="text-xs leading-snug text-ash">The model reviews each piece of code on an agent lane before it runs.</span>
                </div>
                <Switch on={tools.jsSafetyCheck} onChange={(jsSafetyCheck) => onChange({ ...tools, jsSafetyCheck })} labelledBy="tool-js-check-label" />
              </div>
              {!tools.jsSafetyCheck && (
                <span className="border-l-2 border-fault pl-2 text-xs leading-snug text-fault">
                  Off: the code the model writes runs without review.
                </span>
              )}
            </div>
          )}
          <ToolRow
            id="plan"
            label="Plan"
            on={tools.plan}
            onChange={(plan) => onChange({ ...tools, plan })}
            description="The model keeps a checklist of its steps in the reply as it works."
          />
          <ToolRow
            id="memory"
            label="Memory"
            on={tools.memory}
            onChange={(memory) => onChange({ ...tools, memory })}
            description="The model saves short notes in this browser and sees them in every session."
          />
          {tools.memory && <MemorySummary onOpen={onOpenMemory} />}
          <ToolRow
            id="files"
            label="Create files"
            on={tools.files}
            onChange={(files) => onChange({ ...tools, files })}
            description="The model can hand you a text file to download."
          />
          <ToolRow
            id="read-files"
            label="Attachments"
            on={tools.readFiles}
            onChange={(readFiles) => onChange({ ...tools, readFiles })}
            description="Attach text or PDF files to a session; the model reads them in pieces."
          />
        </div>
      )}
    </fieldset>
  );
}

function ToolRow(props: { id: string; label: string; description: string; on: boolean; onChange: (on: boolean) => void }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div className="flex flex-col gap-0.5">
        <span id={`tool-${props.id}-label`} className="font-display text-sm font-semibold text-ink">
          {props.label}
        </span>
        <span className="text-xs leading-snug text-ash">{props.description}</span>
      </div>
      <Switch on={props.on} onChange={props.onChange} labelledBy={`tool-${props.id}-label`} />
    </div>
  );
}

/** How many notes are saved, and the way into the memory sheet. */
function MemorySummary({ onOpen }: { onOpen: () => void }) {
  const notes = useMemoryNotes();
  return (
    <div className="-mt-2 flex items-center justify-between gap-3">
      <span className="text-xs text-ash">
        {notes.length === 0 ? "No notes yet" : `${notes.length} ${notes.length === 1 ? "note" : "notes"} saved`}
      </span>
      <button
        type="button"
        onClick={onOpen}
        className="cut shrink-0 bg-kiln px-3 py-1.5 font-display text-xs font-semibold text-[#eae8e4] [--cut-size:6px] hover:bg-kiln-line"
      >
        Open memory
      </button>
    </div>
  );
}

/** The Tavily key, saved in this browser as it is typed. */
function TavilyKeyField() {
  const key = useTavilyKey();
  return (
    <label className="flex flex-col gap-1.5">
      <span className="font-display text-xs font-medium text-ash">Tavily API key</span>
      <input
        className={`${field} font-display text-xs`}
        type="password"
        name="tavily-key"
        autoComplete="off"
        spellCheck={false}
        placeholder="tvly-…"
        value={key ?? ""}
        onChange={(e) => setTavilyKey(e.target.value)}
      />
      {key ? (
        <span className="text-xs leading-snug text-ash">Kept in this browser, sent only to Tavily.</span>
      ) : (
        <span className="border-l-2 border-ember pl-2 text-xs leading-snug text-ash">
          No key: searches go to search.tiago.zip, then to DuckDuckGo Lite through r.jina.ai if that fails. Keyless, but
          less reliable.
        </span>
      )}
    </label>
  );
}

/** The Markdown switch: replies formatted, or their raw text. */
function MarkdownSetting({ on, onChange }: { on: boolean; onChange: (on: boolean) => void }) {
  return (
    <div className="mt-auto border-t border-line pt-5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-0.5">
          <span id="markdown-label" className="font-display text-sm font-semibold text-ink">
            Markdown in replies
          </span>
          <span className="text-xs leading-snug text-ash">Headings, lists, tables and code, formatted.</span>
        </div>
        <Switch on={on} onChange={onChange} labelledBy="markdown-label" />
      </div>
    </div>
  );
}
