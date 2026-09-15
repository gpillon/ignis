import { REASONING_EFFORTS } from "../api/request.ts";
import { ignisPrompt, setAllTools, type ToolsState } from "../tools/index.ts";
import { setTavilyKey, useTavilyKey } from "../tools/web/tavilyKey.ts";
import { caption, field } from "../ui/classes.ts";
import { Segmented } from "../ui/Segmented.tsx";
import { Slider } from "../ui/Slider.tsx";
import { Switch } from "../ui/Switch.tsx";
import { EFFORT_LABELS, type PlaygroundSettings } from "./defaults.ts";
import { SystemPromptField } from "./SystemPromptField.tsx";

/** The right-hand panel: request settings, tools and display. A drawer below `lg`, shown when `open`. */
export function SettingsPanel(props: {
  open: boolean;
  settings: PlaygroundSettings;
  set: <K extends keyof PlaygroundSettings>(key: K, value: PlaygroundSettings[K]) => void;
  tools: ToolsState;
  onToolsChange: (tools: ToolsState) => void;
  markdown: boolean;
  onMarkdownChange: (on: boolean) => void;
}) {
  const { settings, set } = props;
  // ignis refuses greedy sampling with a top_p it would ignore.
  const greedyConflict = settings.temperature === 0 && settings.topP !== 1;
  return (
    <aside
      aria-label="Settings"
      className={`fixed inset-y-0 right-0 z-40 flex w-72 flex-col gap-6 overflow-y-auto border-l border-line bg-ground px-5 py-5 transition-transform motion-reduce:transition-none lg:static lg:z-auto lg:w-68 lg:translate-x-0 ${props.open ? "translate-x-0" : "max-lg:invisible max-lg:translate-x-full"}`}
    >
      <SystemPromptField value={settings.systemPrompt} onChange={(v) => set("systemPrompt", v)} ignis={ignisPrompt(props.tools)} />

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

      <ToolsSetting tools={props.tools} onChange={props.onToolsChange} />

      <MarkdownSetting on={props.markdown} onChange={props.onMarkdownChange} />
    </aside>
  );
}

function ToolsSetting({ tools, onChange }: { tools: ToolsState; onChange: (tools: ToolsState) => void }) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="sr-only">Tools</legend>
      <div className="mb-2 flex items-center justify-between gap-3">
        <span id="tools-all-label" className={caption}>
          Tools
        </span>
        <Switch on={tools.enabled} onChange={(on) => onChange(setAllTools(tools, on))} labelledBy="tools-all-label" />
      </div>
      {tools.enabled && (
      <div className="flex flex-col gap-2">
        <div className="flex items-start justify-between gap-3">
          <div className="flex flex-col gap-0.5">
            <span id="tool-agents-label" className="font-display text-sm font-semibold text-ink">
              Agents
            </span>
            <span className="text-xs leading-snug text-ash">
              The model can hand sub-tasks to agents that run in parallel on agent lanes. Agents get the other tools
              that are on.
            </span>
          </div>
          <Switch on={tools.agents} onChange={(agents) => onChange({ ...tools, agents })} labelledBy="tool-agents-label" />
        </div>
        <div className="mt-2 flex items-start justify-between gap-3">
          <div className="flex flex-col gap-0.5">
            <span id="tool-web-label" className="font-display text-sm font-semibold text-ink">
              Web
            </span>
            <span className="text-xs leading-snug text-ash">
              The model can search the web and read pages, from this browser.
            </span>
          </div>
          <Switch on={tools.web} onChange={(web) => onChange({ ...tools, web })} labelledBy="tool-web-label" />
        </div>
        {tools.web && <TavilyKeyField />}
      </div>
      )}
    </fieldset>
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
