import { isValidElement, memo, type ReactNode, useDeferredValue, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// Replies rendered as Markdown: CommonMark plus GitHub's tables,
// strikethrough and task lists. Raw HTML in a reply is not rendered, and
// link targets go through react-markdown's URL filter (no `javascript:`).
// Images become links, so a reply never makes the browser fetch anything.

/** The plain text inside rendered children (a code block's source, for Copy). */
export function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) return textOf(node.props.children);
  return "";
}

/** Clipboard write that also works where `navigator.clipboard` is missing (plain http off localhost). */
async function copyText(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.append(area);
  area.select();
  document.execCommand("copy");
  area.remove();
}

function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const className = isValidElement<{ className?: string }>(children) ? (children.props.className ?? "") : "";
  const language = /language-([\w+#-]+)/.exec(className)?.[1];
  const copy = () =>
    void copyText(textOf(children).replace(/\n$/, "")).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  return (
    <div className="md-code cut">
      <div className="md-code-bar">
        <span>{language ?? "code"}</span>
        <button type="button" onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>{children}</pre>
    </div>
  );
}

const components: Components = {
  pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noreferrer noopener">
      {children}
    </a>
  ),
  img: ({ src, alt }) => (
    <a href={typeof src === "string" ? src : undefined} target="_blank" rel="noreferrer noopener">
      {alt || "image"}
    </a>
  ),
  table: ({ children }) => (
    <div className="md-table">
      <table>{children}</table>
    </div>
  ),
};

const plugins = [remarkGfm];

/**
 * A Markdown reply. While it streams, parsing trails the newest tokens a
 * little (a deferred value) so long replies keep the page responsive, and
 * the ember caret sits after the last block.
 */
export const Markdown = memo(function Markdown({ text, streaming = false }: { text: string; streaming?: boolean }) {
  const deferred = useDeferredValue(text);
  return (
    <div className="md" data-streaming={streaming}>
      <ReactMarkdown remarkPlugins={plugins} components={components}>
        {deferred}
      </ReactMarkdown>
    </div>
  );
});
