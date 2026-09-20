// Class strings the settings controls share: a control's caption, and a text field.

export const caption = "font-display text-[13px] font-medium text-ash";

/**
 * A text field with **no width of its own**, for a control that is not the
 * full width of its row.
 *
 * It exists because `field` carries `w-full`, and a second width utility
 * beside it does not win: Tailwind emits its own utilities in its own order,
 * so `` `${field} w-[7.5rem]` `` resolves to `width: 100%` however the class
 * attribute is written. On a `shrink-0` flex item that cannot be given back,
 * which is a row wider than its column and a horizontal scrollbar on the panel
 * around it.
 */
export const fieldLook =
  "rounded-[2px] border border-line bg-surface px-2.5 py-2 text-sm text-ink placeholder:text-ash/70 focus:border-ember focus:outline-none";

export const field = `w-full ${fieldLook}`;
