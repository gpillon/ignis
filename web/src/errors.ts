// A rejected request, readable (GitHub #164): ignis answers 400/404/413/503/
// 504 with OpenAI's `{"error": {message, type, code}}` body.

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
