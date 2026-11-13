// A rejected request, readable (GitHub #164): ignis answers 400/404/413/503/
// 504 with OpenAI's `{"error": {message, type, code}}` body.

/** The error body's `code`, when it names one: what a caller branches on, never the message's words. */
export function apiErrorCode(body: string): string | undefined {
  try {
    const code = (JSON.parse(body) as { error?: { code?: unknown } }).error?.code;
    return typeof code === "string" && code ? code : undefined;
  } catch {
    return undefined;
  }
}

export function apiErrorMessage(status: number, body: string): string {
  try {
    const message = (JSON.parse(body) as { error?: { message?: unknown } }).error?.message;
    if (typeof message === "string" && message) return `${status}: ${message}`;
  } catch {
    // Not JSON: fall through to the raw body.
  }
  const text = body.trim();
  return text ? `${status}: ${text}` : `${status}`;
}
