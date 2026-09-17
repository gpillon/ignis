// The date and time "tool": nothing to call. While it is on, the current day,
// date and time in the browser's time zone go into the prompt — by default the
// moment the session started, written once into the ignis system prompt.
//
// The moment is frozen on purpose. A prompt reuses the engine's state for the
// tokens it shares with the last one, and this block sits ahead of the whole
// conversation: a clock that ticks between two turns changes the prompt's very
// first tokens, nothing is shared, and a long conversation prefills again from
// zero (~25 s at 98K tokens). The cost of freezing is that the time is the
// session's, so it goes stale over hours; the date and time tool's "update
// every prompt" option buys the current moment back by sending it as a
// developer message *after* the history instead, where a change costs nothing.

/** The browser's IANA time zone, `Europe/Rome`. */
export function browserTimeZone(): string {
  return Intl.DateTimeFormat().resolvedOptions().timeZone;
}

/** What the date and time tool adds to the ignis system prompt, for `now`. */
export function dateTimePrompt(now: Date, timeZone = browserTimeZone()): string {
  const when = new Intl.DateTimeFormat("en-GB", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
    timeZone,
  }).format(now);
  const offset =
    new Intl.DateTimeFormat("en-GB", { timeZone, timeZoneName: "longOffset" }).formatToParts(now).find((p) => p.type === "timeZoneName")
      ?.value ?? "";
  return `# Date and time
It is now ${when} in the user's time zone (${timeZone}${offset ? `, ${offset}` : ""}). Use it for anything that depends on today's date or the current time.`;
}
