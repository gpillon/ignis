import { useEffect, useRef, useState } from "react";
import { canChat, type ModelState } from "../api/model.ts";
import { runningFamily } from "../server/band.ts";
import { section } from "../server/form.ts";
import { type CatalogRow, type ConfigDocument, readCatalog, readConfig } from "../server/routes.ts";

// Whether the loaded model has vision (#394), which decides whether a picture
// the model reads in the workspace goes into the tool's output: with vision
// off, any picture in the prompt refuses the whole request (`vision_disabled`).
// The page learns it from what the server reports, the running family's
// `vision.enabled` in `GET /v1/config`, and reads it again after every switch:
// `make_image` switches models in the middle of a turn. Unknown is no vision.
// The Transcribe view follows it as a state: a context image waits to be read
// until the text model has vision (#437). Unloaded while idle, the model has no
// row loaded, but its own row names its family: the page knows its vision then
// too, and a reading loads it again.

/**
 * The family `model` runs as: the loaded row's; while none is loaded (unloaded idle), the family of its own row, by its
 * id, or of the entry it serves from under another id (`start_model`, `known_models`); null when nothing says.
 */
function familyOf(catalog: CatalogRow[] | null, model: string | null): string | null {
  const running = runningFamily(catalog);
  if (running !== null || catalog === null || model === null) return running;
  const own = catalog.find((row) => row.id === model) ?? catalog.find((row) => row.start_model === model || row.known_models?.includes(model) === true);
  return own?.family ?? null;
}

/**
 * The vision of the model loaded, or of `model` while it is unloaded idle: its family's own `vision.enabled`, else the
 * group's; none for no family known.
 */
export function visionOf(config: ConfigDocument | null, catalog: CatalogRow[] | null, model: string | null = null): boolean {
  const family = familyOf(catalog, model);
  if (!config || !family) return false;
  const own = section(config, "vision", family);
  const enabled = own && "enabled" in own ? own.enabled : config.vision?.enabled;
  return enabled === true;
}

/** The vision of the model loaded, or of `model` while it is unloaded idle, as the server reports it now; none when it cannot say. */
export async function readVision(fetcher: typeof fetch = fetch, model: string | null = null): Promise<boolean> {
  const [config, catalog] = await Promise.all([readConfig(fetcher), readCatalog(fetcher)]);
  return config.ok && catalog.ok ? visionOf(config.value, catalog.value, model) : false;
}

/** A read of the vision of `model`, the one the page sees, from the server. */
const readFor = (model: string) => readVision(fetch, model);

/** What the page knows of the vision of the model it chats with, kept up to date with the model it sees loaded. */
export type VisionWatch = {
  /**
   * The model the page can chat with now, by id, or null while there is none
   * (a switch, the image model): a model ready again after anything else is
   * read again. Resolves once the read for it has landed.
   */
  seen(model: string | null): Promise<void>;
  /** Whether that model has vision, as far as its own read says; none until the read lands. */
  has(): boolean;
};

export function createVisionWatch(read: (model: string) => Promise<boolean>): VisionWatch {
  let model: string | null = null;
  let known: { model: string; vision: boolean } | null = null;
  let reading: Promise<void> = Promise.resolve();
  return {
    seen(next) {
      if (next === model) return reading;
      model = next;
      known = null;
      if (next === null) return (reading = Promise.resolve());
      reading = read(next).then(
        (vision) => {
          // An answer for a model no longer loaded says nothing of the one that is.
          if (model === next) known = { model: next, vision };
        },
        () => {},
      );
      return reading;
    },
    has: () => known !== null && known.model === model && known.vision,
  };
}

/**
 * The model whose vision a view follows (#437): the text model the page can chat with, by id, ready or unloaded idle,
 * whose own row in the catalog says then; null with none.
 */
export function visionModel(model: ModelState): string | null {
  return canChat(model) ? model.id : null;
}

/**
 * Whether the text model has vision, as a state a view redraws on (the Transcribe view's context images, #437): read
 * each time the page sees it after anything else, unloaded idle too, so a page loaded meanwhile knows; none until a read
 * for it lands.
 */
export function useHasVision(model: ModelState, read: (model: string) => Promise<boolean> = readFor): boolean {
  const [watch] = useState(() => createVisionWatch(read));
  const [known, setKnown] = useState<{ model: string; vision: boolean } | null>(null);
  const id = visionModel(model);
  useEffect(() => {
    if (id === null) {
      void watch.seen(null);
      setKnown(null);
      return;
    }
    let current = true;
    void watch.seen(id).then(() => current && setKnown({ model: id, vision: watch.has() }));
    return () => {
      current = false;
    };
  }, [watch, id]);
  return id !== null && known?.model === id && known.vision;
}

/**
 * The page's watch on the vision of the model it chats with: read whenever
 * that model changes, and asked at the moment a tool needs it.
 */
export function useVision(model: ModelState, read: (model: string) => Promise<boolean> = readFor): () => boolean {
  const watch = useRef<VisionWatch | null>(null);
  watch.current ??= createVisionWatch(read);
  const chatting = canChat(model) && model.state === "ready" ? model.id : null;
  useEffect(() => {
    void watch.current?.seen(chatting);
  }, [chatting]);
  return watch.current.has;
}
