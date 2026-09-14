/** How far along its track a slider's value sits, as a CSS percentage (clamped to the track). */
export function trackFill(value: number, min: number, max: number): string {
  if (max <= min) return "0%";
  const share = Math.min(1, Math.max(0, (value - min) / (max - min)));
  return `${share * 100}%`;
}
