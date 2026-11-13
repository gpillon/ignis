import { describe, expect, it } from "vitest";
import { apiErrorCode, apiErrorMessage } from "./errors.ts";

describe("apiErrorCode", () => {
  it("reads the code an OpenAI error names, and nothing else", () => {
    const body = (code: unknown) => JSON.stringify({ error: { message: "m", type: "invalid_request_error", code } });
    expect(apiErrorCode(body("model_unsupported"))).toBe("model_unsupported");
    expect(apiErrorCode(body(null))).toBeUndefined();
    expect(apiErrorCode("Bad Gateway")).toBeUndefined();
  });
});

describe("apiErrorMessage", () => {
  it("shows the OpenAI error message with its status", () => {
    const body = JSON.stringify({ error: { message: "the engine cannot admit the request right now", type: "server_error", code: "engine_full" } });
    expect(apiErrorMessage(503, body)).toBe("503: the engine cannot admit the request right now");
  });

  it("falls back to the raw body when it is not an OpenAI error", () => {
    expect(apiErrorMessage(502, "Bad Gateway")).toBe("502: Bad Gateway");
  });

  it("names the status alone when there is no body", () => {
    expect(apiErrorMessage(504, "")).toBe("504");
  });
});
