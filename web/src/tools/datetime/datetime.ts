// The date and time "tool": nothing to call. While it is on, every request's
// ignis system prompt starts with the current day, date and time in the
// browser's time zone, taken when the turn starts.

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
