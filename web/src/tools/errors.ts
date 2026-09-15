/** The error a call to a tool the request did not declare gets back. */
export function unknownToolError(name: string, available: string[]): string {
  if (available.length === 0) return `Unknown tool "${name}": no tools are available.`;
  return `Unknown tool "${name}": the tools available are ${available.map((n) => `"${n}"`).join(", ")}.`;
}
