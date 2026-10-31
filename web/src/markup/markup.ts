// Marking a picture: the markup editor's geometry and what is made of the
// marks, shared by the image view and the chat. Qwen-Image-2.1's card says a
// local edit is asked "via circles, painted annotations, or separate masks":
// the owner paints strokes or draws ellipses on a picture, and a request
// carries them one of two ways —
//   - drawn on the picture: the marks are baked into the picture it sends
//     (the chat's only way: the text model sees the marks too);
//   - as a mask: the picture goes clean, followed by a mask image of its own
//     (the marked area white on black), which the prompt names by number.
// Either way the result can be kept to the marked area after the fact: the
// page lays it over the original through the area, feathered, so nothing
// outside it changes (`keepOutside`). The edits route takes no `mask` field
// (spec image-model/05): the mask is one more condition image, as the card
// has it.
//
// Marks live in the picture's own pixels, whatever size it is shown at. The
// geometry is pure and tested here; the canvas work at the bottom runs only
// in the browser.

export type Point = [number, number];

export type Mark =
  | { kind: "stroke"; color: string; width: number; points: Point[] }
  | { kind: "ellipse"; color: string; width: number; from: Point; to: Point };

export type MarkTool = Mark["kind"];

/** How the marks reach the model: drawn on the picture, or as a mask image after it. */
export type MarkMode = "drawn" | "mask";

/**
 * The colours a mark can take; red first, as an edit request usually circles in red. Black and white for pictures
 * where red would not stand out, or where the prompt wants a plain outline.
 */
export const MARK_COLORS = [
  { value: "#ff2a2a", name: "red" },
  { value: "#ffd400", name: "yellow" },
  { value: "#21d07a", name: "green" },
  { value: "#ffffff", name: "white" },
  { value: "#000000", name: "black" },
] as const;

export const colorName = (color: string) => MARK_COLORS.find((c) => c.value === color)?.name ?? "the marked colour";

/** What saving the marks made, at the picture's own size. */
export type MarkedPicture = {
  marks: Mark[];
  mode: MarkMode;
  keepOutside: boolean;
  /** The picture with the marks drawn on it. */
  drawn: Blob;
  /** The marked area, white on black. */
  mask: Blob;
  /** The marked area as alpha, for laying a result over the original. */
  region: Blob;
  width: number;
  height: number;
};

/** A point on the shown picture, in the picture's own pixels. */
export function toImagePoint(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  natural: { width: number; height: number },
): Point {
  const x = ((clientX - rect.left) / rect.width) * natural.width;
  const y = ((clientY - rect.top) / rect.height) * natural.height;
  return [Math.min(natural.width, Math.max(0, x)), Math.min(natural.height, Math.max(0, y))];
}

/** A stroke takes a new point only once the pointer has moved: a still hand adds nothing. */
export function extendStroke(points: Point[], next: Point, minStep: number): Point[] {
  const last = points[points.length - 1];
  if (last && Math.hypot(next[0] - last[0], next[1] - last[1]) < minStep) return points;
  return [...points, next];
}

/** An ellipse's centre and radii, from the two corners of the drag that drew it. */
export function ellipseOf(from: Point, to: Point): { cx: number; cy: number; rx: number; ry: number } {
  return { cx: (from[0] + to[0]) / 2, cy: (from[1] + to[1]) / 2, rx: Math.abs(to[0] - from[0]) / 2, ry: Math.abs(to[1] - from[1]) / 2 };
}

/** A mark worth keeping: a stroke with any length, an ellipse with some size to it. */
export function isMark(mark: Mark): boolean {
  if (mark.kind === "stroke") return mark.points.length > 0;
  const { rx, ry } = ellipseOf(mark.from, mark.to);
  return rx >= 2 && ry >= 2;
}

/** The feather of the edge where a kept result meets the original: about half a percent of the short side. */
export const featherOf = (width: number, height: number) => Math.max(2, Math.round(Math.min(width, height) * 0.006));

/** A start for the prompt that names the marks the way they were sent; the owner finishes the sentence. */
export function promptStub(mode: MarkMode, numbered: { picture: number; mask?: number }, color: string): string {
  if (mode === "mask" && numbered.mask !== undefined) return `In image ${numbered.picture}, change only the area that is white in image ${numbered.mask}: `;
  return `In image ${numbered.picture}, change what is marked in ${colorName(color)}: `;
}

// A context image's editing (spec #428), the editor's other two tools: a crop
// keeps one area of the picture, a black-out paints one solid black. Each is a
// rectangle in the picture's own pixels, and only the picture they make is
// kept: the original, and what is under a black-out, never are.

/** A rectangle in the picture's own pixels. */
export type Rect = { x: number; y: number; width: number; height: number };

export type Redaction = { kind: "crop" | "blackout"; rect: Rect };

export type RedactTool = Redaction["kind"];

/**
 * The rectangle a drag draws, from either corner, held inside the picture and widened to whole pixels: a black-out
 * leaves no part of a pixel showing at its edge.
 */
export function rectOf(from: Point, to: Point, natural: { width: number; height: number }): Rect {
  const inside = (v: number, max: number) => Math.min(max, Math.max(0, v));
  const x = inside(Math.floor(Math.min(from[0], to[0])), natural.width);
  const y = inside(Math.floor(Math.min(from[1], to[1])), natural.height);
  const right = inside(Math.ceil(Math.max(from[0], to[0])), natural.width);
  const bottom = inside(Math.ceil(Math.max(from[1], to[1])), natural.height);
  return { x, y, width: right - x, height: bottom - y };
}

/** A rectangle worth keeping: a drag, not a click. */
export const isArea = (rect: Rect): boolean => rect.width >= 2 && rect.height >= 2;

/**
 * What the edits make of a picture: the area kept, the last crop drawn or the whole picture, and the black-outs in
 * the kept picture's own pixels, each cut at its edges and one outside it dropped.
 */
export function redactionPlan(edits: Redaction[], natural: { width: number; height: number }): { crop: Rect; blackouts: Rect[] } {
  const crops = edits.filter((e) => e.kind === "crop");
  const crop = crops[crops.length - 1]?.rect ?? { x: 0, y: 0, width: natural.width, height: natural.height };
  const blackouts: Rect[] = [];
  for (const { kind, rect } of edits) {
    if (kind !== "blackout") continue;
    const x = Math.max(rect.x, crop.x);
    const y = Math.max(rect.y, crop.y);
    const right = Math.min(rect.x + rect.width, crop.x + crop.width);
    const bottom = Math.min(rect.y + rect.height, crop.y + crop.height);
    if (right > x && bottom > y) blackouts.push({ x: x - crop.x, y: y - crop.y, width: right - x, height: bottom - y });
  }
  return { crop, blackouts };
}

// ---------------------------------------------------------------------------
// The canvas work: browser only.

function canvasOf(width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("this browser gives no 2D canvas");
  return { canvas, ctx };
}

/** Draws the marks as they look, `scale` being shown pixels per picture pixel. */
export function drawMarks(ctx: CanvasRenderingContext2D, marks: Mark[], scale = 1) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const mark of marks) {
    ctx.strokeStyle = mark.color;
    ctx.lineWidth = mark.width * scale;
    ctx.beginPath();
    if (mark.kind === "stroke") {
      const [first, ...rest] = mark.points;
      ctx.moveTo(first[0] * scale, first[1] * scale);
      // One point is a dot: a zero-length segment with round caps.
      if (rest.length === 0) ctx.lineTo(first[0] * scale + 0.01, first[1] * scale);
      for (const p of rest) ctx.lineTo(p[0] * scale, p[1] * scale);
    } else {
      const e = ellipseOf(mark.from, mark.to);
      ctx.ellipse(e.cx * scale, e.cy * scale, e.rx * scale, e.ry * scale, 0, 0, Math.PI * 2);
    }
    ctx.stroke();
  }
  ctx.restore();
}

/** Draws the marked area in white: a stroke as wide as it was painted, an ellipse filled. */
function drawArea(ctx: CanvasRenderingContext2D, marks: Mark[]) {
  ctx.save();
  ctx.fillStyle = "#fff";
  drawMarks(ctx, marks.map((m) => ({ ...m, color: "#fff" })));
  for (const mark of marks) {
    if (mark.kind !== "ellipse") continue;
    const e = ellipseOf(mark.from, mark.to);
    ctx.beginPath();
    ctx.ellipse(e.cx, e.cy, e.rx, e.ry, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

const toBlob = (canvas: HTMLCanvasElement) =>
  new Promise<Blob>((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("the canvas gave no PNG"))), "image/png"));

export const loadImage = (url: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("the picture could not be read"));
    image.src = url;
  });

/** What saving the marks makes of the picture at `sourceUrl`: drawn on, its mask, and its area. */
export async function renderMarks(sourceUrl: string, marks: Mark[], mode: MarkMode, keepOutside: boolean): Promise<MarkedPicture> {
  const picture = await loadImage(sourceUrl);
  const width = picture.naturalWidth;
  const height = picture.naturalHeight;

  const drawn = canvasOf(width, height);
  drawn.ctx.drawImage(picture, 0, 0);
  drawMarks(drawn.ctx, marks);

  const mask = canvasOf(width, height);
  mask.ctx.fillStyle = "#000";
  mask.ctx.fillRect(0, 0, width, height);
  drawArea(mask.ctx, marks);

  const region = canvasOf(width, height);
  drawArea(region.ctx, marks);

  const [drawnBlob, maskBlob, regionBlob] = await Promise.all([toBlob(drawn.canvas), toBlob(mask.canvas), toBlob(region.canvas)]);
  return { marks, mode, keepOutside, drawn: drawnBlob, mask: maskBlob, region: regionBlob, width, height };
}

/**
 * A result laid over the original through the marked area, feathered at its edge: outside it, the original's own
 * pixels. The result is scaled to the original's size, so the picture keeps the size it was given at.
 */
export async function keepOutside(result: Blob, keep: { source: Blob; region: Blob }): Promise<Blob> {
  const urls = [result, keep.source, keep.region].map((b) => URL.createObjectURL(b));
  try {
    const [made, original, region] = await Promise.all(urls.map(loadImage));
    const width = original.naturalWidth;
    const height = original.naturalHeight;
    const layer = canvasOf(width, height);
    layer.ctx.drawImage(made, 0, 0, width, height);
    layer.ctx.globalCompositeOperation = "destination-in";
    layer.ctx.filter = `blur(${featherOf(width, height)}px)`;
    layer.ctx.drawImage(region, 0, 0, width, height);
    const out = canvasOf(width, height);
    out.ctx.drawImage(original, 0, 0);
    out.ctx.drawImage(layer.canvas, 0, 0);
    return await toBlob(out.canvas);
  } finally {
    for (const url of urls) URL.revokeObjectURL(url);
  }
}

/** The kiln's ground behind what a crop leaves out, and the crop's frame: the editor's own colours. */
const CROP_DIM = "rgba(28, 32, 38, 0.72)";
const CROP_FRAME = "#ff5a1f";

/**
 * Draws the edits as they will come out, `scale` being shown pixels per picture pixel: the black-outs opaque, and
 * what the crop leaves out dimmed behind its frame.
 */
export function drawRedactions(ctx: CanvasRenderingContext2D, edits: Redaction[], natural: { width: number; height: number }, scale = 1) {
  const { crop } = redactionPlan(edits, natural);
  ctx.save();
  ctx.fillStyle = "#000";
  for (const { kind, rect } of edits) if (kind === "blackout") ctx.fillRect(rect.x * scale, rect.y * scale, rect.width * scale, rect.height * scale);
  if (crop.width < natural.width || crop.height < natural.height) {
    ctx.fillStyle = CROP_DIM;
    ctx.beginPath();
    ctx.rect(0, 0, natural.width * scale, natural.height * scale);
    ctx.rect(crop.x * scale, crop.y * scale, crop.width * scale, crop.height * scale);
    ctx.fill("evenodd");
    ctx.strokeStyle = CROP_FRAME;
    ctx.lineWidth = 2;
    ctx.strokeRect(crop.x * scale, crop.y * scale, crop.width * scale, crop.height * scale);
  }
  ctx.restore();
}

/** A picture as its edits made it, at its own size. */
export type RedactedPicture = { blob: Blob; width: number; height: number };

/**
 * The picture at `sourceUrl` as its edits make it (`redactionPlan`): the area kept, and the black-outs painted
 * opaque into its pixels, as a PNG. Drawn even with no edit, so what is kept is never the picked file's own bytes,
 * nor what the file carries beside its pixels.
 */
export async function renderRedactions(sourceUrl: string, edits: Redaction[]): Promise<RedactedPicture> {
  const picture = await loadImage(sourceUrl);
  const { crop, blackouts } = redactionPlan(edits, { width: picture.naturalWidth, height: picture.naturalHeight });
  const out = canvasOf(crop.width, crop.height);
  out.ctx.drawImage(picture, crop.x, crop.y, crop.width, crop.height, 0, 0, crop.width, crop.height);
  out.ctx.fillStyle = "#000";
  for (const r of blackouts) out.ctx.fillRect(r.x, r.y, r.width, r.height);
  return { blob: await toBlob(out.canvas), width: crop.width, height: crop.height };
}
