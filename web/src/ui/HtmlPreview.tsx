import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { IconClose } from "./icons.tsx";

// HTML the model wrote, shown two ways: rendered in a sandboxed iframe, or as
// its source. The frame runs scripts but gets an opaque origin (no
// allow-same-origin), so the page cannot reach the Playground, its storage or
// its API key, and it cannot navigate the top window or open popups. Any
// preview opens full screen, at a desktop, tablet or phone width.

export const isHtmlName = (name: string) => /\.html?$/i.test(name);

const escapeHtml = (text: string) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** The page with a <title> (which names the PDF), taken from the file name when it has none. */
export function withPrintTitle(html: string, name: string): string {
  if (/<title[\s>]/i.test(html)) return html;
  const title = `<title>${escapeHtml(name.replace(/\.html?$/i, "") || "page")}</title>`;
  return /<head[^>]*>/i.test(html) ? html.replace(/<head[^>]*>/i, (head) => `${head}${title}`) : `${title}${html}`;
}

/**
 * Opens the browser's print dialog on the page alone, where "Save as PDF"
 * writes the file. The page loads in a hidden frame with the same sandbox
 * plus modals (print needs them), runs its scripts, prints itself, and the
 * frame goes once the dialog closes.
 */
export function saveAsPdf(html: string, name: string) {
  const frame = document.createElement("iframe");
  frame.setAttribute("sandbox", `${SANDBOX} allow-modals`);
  frame.setAttribute("aria-hidden", "true");
  frame.style.cssText = "position:fixed;left:-10000px;top:0;width:1024px;height:768px;border:0";
  const cleanup = () => {
    window.removeEventListener("message", done);
    frame.remove();
  };
  const done = (event: MessageEvent) => {
    if (event.source === frame.contentWindow && event.data === "ignis-printed") cleanup();
  };
  window.addEventListener("message", done);
  setTimeout(cleanup, 10 * 60_000);
  frame.srcdoc = `${withPrintTitle(html, name)}<script>addEventListener("load",()=>setTimeout(()=>{print();parent.postMessage("ignis-printed","*")},300));</script>`;
  document.body.append(frame);
}

const PDF_HINT = "Opens the print dialog for this page: choose Save as PDF";

const SANDBOX = "allow-scripts allow-forms";

const WIDTHS = [
  { id: "full", label: "Full width", px: null },
  { id: "tablet", label: "Tablet", px: 768 },
  { id: "phone", label: "Phone", px: 390 },
] as const;

const FRAME_BUTTON =
  "cut bg-kiln/85 px-2.5 py-1 font-display text-xs font-semibold text-[#eae8e4] [--cut-size:6px] hover:bg-kiln";

/** The rendered page in a sandboxed frame; drag its bottom edge to resize, expand it, or save it as PDF. */
export function HtmlFrame({ html, title }: { html: string; title: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="relative">
      <div className="h-96 min-h-40 resize-y overflow-hidden bg-white">
        <iframe title={title} sandbox={SANDBOX} srcDoc={html} className="block size-full border-0 bg-white" />
      </div>
      <div className="absolute top-2 right-2 flex gap-1.5">
        <button type="button" title={PDF_HINT} onClick={() => saveAsPdf(html, title)} className={FRAME_BUTTON}>
          Save as PDF
        </button>
        <button type="button" onClick={() => setExpanded(true)} className={FRAME_BUTTON}>
          Expand preview
        </button>
      </div>
      {expanded && <HtmlFullscreen html={html} title={title} onClose={() => setExpanded(false)} />}
    </div>
  );
}

/**
 * The preview over the whole window, in a portal: the reply's chamfered
 * boxes clip fixed children, so the sheet lives on the body.
 */
function HtmlFullscreen({ html, title, onClose }: { html: string; title: string; onClose: () => void }) {
  const [width, setWidth] = useState<(typeof WIDTHS)[number]["id"]>("full");
  const px = WIDTHS.find((w) => w.id === width)?.px ?? null;

  useEffect(() => {
    const close = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);

  return createPortal(
    <div role="dialog" aria-modal="true" aria-label={title} className="memory-sheet fixed inset-0 z-[60] flex flex-col bg-kiln text-[#eae8e4]">
      <header className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-kiln-line px-4 py-2.5 md:px-6">
        <h2 className="min-w-0 flex-1 truncate font-display text-[15px] font-semibold">{title}</h2>
        <div role="radiogroup" aria-label="Preview width" className="flex gap-4 font-display text-[13px] font-medium">
          {WIDTHS.map((w) => (
            <button
              key={w.id}
              type="button"
              role="radio"
              aria-checked={width === w.id}
              onClick={() => setWidth(w.id)}
              className={`border-b-2 pb-0.5 ${width === w.id ? "border-[#ff5a1f] text-[#eae8e4]" : "border-transparent text-[#939ba4] hover:text-[#eae8e4]"}`}
            >
              {w.label}
            </button>
          ))}
        </div>
        <button
          type="button"
          title={PDF_HINT}
          onClick={() => saveAsPdf(html, title)}
          className="cut bg-[#ff5a1f] px-3 py-1.5 font-display text-xs font-semibold text-[#1c2026] [--cut-size:6px] hover:bg-[#ff7a45]"
        >
          Save as PDF
        </button>
        <button
          type="button"
          aria-label="Close the preview"
          title="Close (Esc)"
          onClick={onClose}
          className="-mr-2 grid size-9 place-items-center text-[#939ba4] hover:bg-kiln-line hover:text-white"
        >
          <IconClose />
        </button>
      </header>
      <div className="flex min-h-0 flex-1 justify-center overflow-auto bg-[#12151a] p-3 md:p-5">
        <iframe
          title={title}
          sandbox={SANDBOX}
          srcDoc={html}
          style={px ? { width: px } : undefined}
          className={`block h-full shrink-0 border-0 bg-white shadow-[0_24px_60px_rgb(0_0_0/0.45)] ${px ? "max-w-none" : "w-full"}`}
        />
      </div>
    </div>,
    document.body,
  );
}

/** A tab strip over the preview or the source; the preview comes first. */
export function HtmlPreview({ html, name }: { html: string; name: string }) {
  const [tab, setTab] = useState<"preview" | "source">("preview");
  return (
    <div className="flex flex-col gap-2">
      <div role="tablist" aria-label={`How to show ${name}`} className="flex gap-4 font-display text-xs font-semibold">
        {(["preview", "source"] as const).map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={tab === t}
            onClick={() => setTab(t)}
            className={`border-b-2 pb-0.5 ${tab === t ? "border-ember text-ink" : "border-transparent text-ash hover:text-ink"}`}
          >
            {t === "preview" ? "Preview" : "Source"}
          </button>
        ))}
      </div>
      {tab === "preview" ? (
        <HtmlFrame html={html} title={name} />
      ) : (
        <pre className="max-h-96 overflow-auto bg-ground px-2.5 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words text-ink">
          {html}
        </pre>
      )}
    </div>
  );
}
