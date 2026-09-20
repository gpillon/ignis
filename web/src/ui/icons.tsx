const icon = { width: 16, height: 16, viewBox: "0 0 16 16", fill: "none", stroke: "currentColor", strokeWidth: 1.5, "aria-hidden": true } as const;

export function IconPencil() {
  return (
    <svg {...icon} width={13} height={13}>
      <path d="M10.5 2.5l3 3L6 13H3v-3z" />
    </svg>
  );
}

export function IconFork() {
  return (
    <svg {...icon} width={13} height={13}>
      <circle cx="4" cy="3.5" r="1.5" />
      <circle cx="12" cy="3.5" r="1.5" />
      <circle cx="8" cy="12.5" r="1.5" />
      <path d="M4 5v1c0 1.7 1.3 3 3 3h2c1.7 0 3-1.3 3-3V5M8 9v2" />
    </svg>
  );
}

export function IconRegenerate() {
  return (
    <svg {...icon} width={13} height={13}>
      <path d="M13 8a5 5 0 1 1-1.46-3.54M13.5 2v3h-3" />
    </svg>
  );
}

export function IconPlus() {
  return (
    <svg {...icon}>
      <path d="M8 3v10M3 8h10" />
    </svg>
  );
}

export function IconPaperclip() {
  return (
    <svg {...icon} width={18} height={18}>
      <path d="M13.5 7.5l-5.3 5.3a3.2 3.2 0 0 1-4.5-4.5l5.6-5.6a2.1 2.1 0 0 1 3 3L6.8 11.2a1 1 0 0 1-1.5-1.5l4.9-4.9" />
    </svg>
  );
}

export function IconImage() {
  return (
    <svg {...icon} width={18} height={18}>
      <rect x="2" y="3" width="14" height="11" rx="1.5" />
      <circle cx="6.2" cy="6.8" r="1.2" />
      <path d="M2.6 12.2l3.6-3.4 2.6 2.4 2.4-2.2 4 3.6" />
    </svg>
  );
}

export function IconClose() {
  return (
    <svg {...icon} width={14} height={14}>
      <path d="M4 4l8 8M12 4l-8 8" />
    </svg>
  );
}

/** Delete, said the way every other interface says it. */
export function IconTrash() {
  return (
    <svg {...icon} width={15} height={15}>
      <path d="M3 5h10M6.5 5V3.5h3V5M4.5 5l.6 8.5h5.8L11.5 5M6.8 7.5v4M9.2 7.5v4" />
    </svg>
  );
}

/** Open the picture at its own size. */
export function IconExpand() {
  return (
    <svg {...icon} width={14} height={14}>
      <path d="M6 2.5H2.5V6M10 13.5h3.5V10" />
    </svg>
  );
}

export function IconChevron({ className }: { className?: string }) {
  return (
    <svg {...icon} className={className}>
      <path d="M4 10l4-4 4 4" />
    </svg>
  );
}

export function IconSessions() {
  return (
    <svg {...icon} width={18} height={18}>
      <path d="M2.5 4h11M2.5 8h11M2.5 12h7" />
    </svg>
  );
}

export function IconSliders() {
  return (
    <svg {...icon} width={18} height={18}>
      <path d="M2.5 4.5h6M11.5 4.5h2M2.5 11.5h2M7.5 11.5h6" />
      <path d="M8.5 3v3M4.5 10v3" />
    </svg>
  );
}

export function IconPause() {
  return (
    <svg {...icon} width={12} height={12}>
      <path d="M5 3v10M11 3v10" strokeWidth={2} />
    </svg>
  );
}

export function IconPlay() {
  return (
    <svg {...icon} width={12} height={12}>
      <path d="M4.5 2.5v11l9-5.5z" fill="currentColor" />
    </svg>
  );
}

/** A health verdict's mark, so its colour never carries the meaning alone. */
export function IconHealth({ level }: { level: "idle" | "healthy" | "busy" | "saturated" }) {
  return (
    <svg {...icon} width={22} height={22} strokeWidth={1.8}>
      {level === "idle" && <path d="M3 8h2.5M10.5 8H13M8 3v2.5M8 10.5V13" />}
      {level === "healthy" && <path d="M3 8.5l3.2 3L13 4.5" />}
      {level === "busy" && <path d="M2.5 11a5.5 5.5 0 0 1 11 0M8 11l2.6-3.4" />}
      {level === "saturated" && <path d="M8 2.5l6 11H2zM8 6.5v3.2M8 11.5v.2" />}
    </svg>
  );
}
