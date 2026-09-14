// Chat autoscroll (GitHub #164): follow new tokens only while the reader is
// already at the bottom; once they scroll up to read, leave them there.

/** How far above the true bottom still counts as "at the bottom", in px. */
const SLACK_PX = 40;

export function isAtBottom(view: { scrollTop: number; clientHeight: number; scrollHeight: number }): boolean {
  return view.scrollHeight - view.scrollTop - view.clientHeight <= SLACK_PX;
}
