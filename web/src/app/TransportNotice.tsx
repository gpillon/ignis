import { dismissTransportNotice, useTransport } from "../api/transport.ts";

/**
 * Said once, when the socket could not be opened and the page moved to HTTP
 * (GitHub #283): nothing failed, but the conversation is on the other wire.
 */
export function TransportNotice() {
  const { notice } = useTransport();
  if (!notice) return null;
  return (
    <div role="status" className="flex shrink-0 items-start justify-between gap-4 border-b border-line border-l-2 border-l-ember bg-surface py-2 pl-4 pr-2">
      <div>
        <h2 className="font-display text-[14px] font-semibold text-ink">The conversation is on HTTP</h2>
        <p className="mt-1 text-[13px] leading-snug text-ash">
          ignis did not open the WebSocket, so this page sends its turns as chat completions, and the browser carries five
          streams at once. Picking WebSocket in General settings tries it again.
        </p>
      </div>
      <button type="button" onClick={dismissTransportNotice} className="shrink-0 px-2 py-1 font-display text-[12px] text-ash hover:text-ink">
        Dismiss
      </button>
    </div>
  );
}
