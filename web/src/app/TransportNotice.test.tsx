import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";
import { chooseTransport, dismissTransportNotice, fallBackToHttp } from "../api/transport.ts";
import { TransportNotice } from "./TransportNotice.tsx";

// The one-time word that the page moved to HTTP on its own (GitHub #283).

describe("TransportNotice", () => {
  afterEach(() => chooseTransport("websocket"));

  it("says once that the conversation moved to HTTP, until dismissed", () => {
    expect(renderToStaticMarkup(<TransportNotice />)).toBe("");
    fallBackToHttp();
    const notice = renderToStaticMarkup(<TransportNotice />);
    expect(notice).toContain("The conversation is on HTTP");
    // Picking WebSocket again tries the socket, so the notice does not promise HTTP until a reload.
    expect(notice).toContain("Picking WebSocket in General settings tries it again");
    expect(notice).not.toContain("until it reloads");
    dismissTransportNotice();
    expect(renderToStaticMarkup(<TransportNotice />)).toBe("");
    fallBackToHttp();
    expect(renderToStaticMarkup(<TransportNotice />)).toBe("");
  });
});
