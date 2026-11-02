import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { canSend, type LastSwitch, loadUnderWay, type ModelState } from "../api/model.ts";
import { dropDraft, type Draft, editDraft, type InForce, patchBody, removeDraft } from "./form.ts";
import { afterPatch, afterSwitch, type Line, type PendingReload, settleReload } from "./outcome.ts";
import {
  type Answer,
  cancelDownload,
  type CatalogDisk,
  type CatalogList,
  type CatalogRow,
  type ConfigDocument,
  type FieldDecl,
  type OriginRow,
  patchConfig,
  readCatalog,
  readConfig,
  readFields,
  readOrigins,
  type Refusal,
  removeModel,
  startDownload,
  switchTo,
} from "./routes.ts";
import { afterDownload, followDownloads, nextProgress, type Progress, runningDownload, speedOf } from "./download.ts";
import { afterRemoval } from "./removal.ts";

// The Server view's state (spec playground/05): the declarations, the
// running configuration, where its values come from and the catalog as read, the draft of changes, and
// the answers to "Apply" and "Load". The view renders it and nothing else,
// so each of its states is a plain value a test can render.
//
// The model state is the page's (api/model.ts), which re-reads it each
// second while a switch runs. This hook only starts that following after a
// 202 — the state still reads `ready` until the next read — and reads the
// configuration and the catalog again once the model serves again. While a
// download runs (GitHub #376), whoever started it, the hook reads the catalog
// each second the view is shown, and the speed from those reads.

export type Loaded<T> = { state: "loading" } | { state: "ok"; value: T } | { state: "error"; refusal: Refusal };

export type ServerState = {
  fields: Loaded<FieldDecl[]>;
  config: Loaded<ConfigDocument>;
  /** Where each value of `config` comes from; unread, the rows fall back to the file's shape. */
  origins: Loaded<OriginRow[]>;
  catalog: Loaded<CatalogRow[]>;
  /** Where the server downloads models and the room left there, from the last catalog read that said it (GitHub #453). */
  disk: CatalogDisk | null;
  draft: Draft;
  /** The file keys of the rows the last refusal named. */
  marked: string[];
  /** The dock's lines: the answer to the last "Apply". */
  result: Line[] | null;
  /** A reload a `202` began, followed until it ends. */
  pending: PendingReload | null;
  applying: boolean;
  /** The id whose "Load" is being sent. */
  loadingId: string | null;
  /** Why the server refused the last "Load". */
  loadRefusal: string | null;
  /** The id whose "Download" is being sent. */
  downloadingId: string | null;
  /** Why the server refused the last "Download", "Resume" or "Cancel". */
  downloadRefusal: string | null;
  /** The running download's speed, in bytes per second, once two catalog reads gave one. */
  downloadSpeed: number | null;
  /** A "Cancel" of the running download is being sent (GitHub #377). */
  cancelling: boolean;
  /** The id whose "Remove" waits for its confirmation (GitHub #380). */
  confirmingRemoval: string | null;
  /** The id whose removal is being sent. */
  removingId: string | null;
  /** The band's line after the last removal: what it freed, or why the server refused it. */
  removal: Line | null;
};

export type ServerActions = {
  load: (id: string) => void;
  /** Download the catalog entry `id` onto the server, and follow it; on a partial entry, its Resume. */
  download: (id: string) => void;
  /** Stop the running download, keeping what it fetched (GitHub #377). */
  cancel: () => void;
  /** Ask to confirm the removal of the catalog entry `id`'s files (GitHub #380). */
  askRemoval: (id: string) => void;
  /** Remove them, confirmed: the server deletes them, and the catalog is read again. */
  confirmRemoval: (id: string) => void;
  /** Leave the entry as it is. */
  keepModel: () => void;
  /** Set `decl` where `inForce` says: a family's section, or the group. */
  edit: (decl: FieldDecl, inForce: InForce, value: string | boolean) => void;
  /** Take the family's own value of `decl` away, so the shared one applies to it again. */
  remove: (decl: FieldDecl, scope: string) => void;
  /** Take a pending removal back. */
  undo: (decl: FieldDecl, scope: string) => void;
  discard: () => void;
  apply: () => void;
  dismiss: () => void;
};

const loaded = <T,>(answer: Answer<T>): Loaded<T> => (answer.ok ? { state: "ok", value: answer.value } : { state: "error", refusal: answer });

const sameSwitch = (a: LastSwitch | undefined, b: LastSwitch | undefined) => JSON.stringify(a) === JSON.stringify(b);

export function useServer(model: ModelState, refreshModel: () => void, visible: boolean): [ServerState, ServerActions] {
  const [fields, setFields] = useState<Loaded<FieldDecl[]>>({ state: "loading" });
  const [config, setConfig] = useState<Loaded<ConfigDocument>>({ state: "loading" });
  const [origins, setOrigins] = useState<Loaded<OriginRow[]>>({ state: "loading" });
  const [catalog, setCatalog] = useState<Loaded<CatalogRow[]>>({ state: "loading" });
  const [disk, setDisk] = useState<CatalogDisk | null>(null);
  const [draft, setDraft] = useState<Draft>({});
  const [marked, setMarked] = useState<string[]>([]);
  const [result, setResult] = useState<Line[] | null>(null);
  const [pending, setPending] = useState<PendingReload | null>(null);
  const [applying, setApplying] = useState(false);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [loadRefusal, setLoadRefusal] = useState<string | null>(null);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [downloadRefusal, setDownloadRefusal] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [confirmingRemoval, setConfirmingRemoval] = useState<string | null>(null);
  const [removingId, setRemovingId] = useState<string | null>(null);
  const [removal, setRemoval] = useState<Line | null>(null);
  // A download runs: the catalog is read each second, the speed taken from the reads.
  const [following, setFollowing] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  // A 202 answered: follow the model until it leaves `ready` or records another switch.
  const [awaiting, setAwaiting] = useState<{ before?: LastSwitch } | null>(null);
  const modelNow = useRef(model);
  modelNow.current = model;

  // A catalog read: the rows, and whether a download runs to be followed.
  const acceptCatalog = useCallback((answer: Answer<CatalogList>) => {
    setCatalog(loaded(answer));
    if (!answer.ok) return;
    setDisk(answer.value.disk ?? null);
    setProgress((current) => nextProgress(current, answer.value, Date.now()));
    setFollowing(runningDownload(answer.value) !== null);
  }, []);

  // The declarations never change while the server runs: read until one read
  // answers (the warm-up holds every /v1 route but GET /v1/models).
  const fieldsRead = useRef(false);
  const reread = useCallback(() => {
    if (!fieldsRead.current) {
      void readFields().then((answer) => {
        fieldsRead.current = answer.ok;
        setFields(loaded(answer));
      });
    }
    void readConfig().then((answer) => setConfig(loaded(answer)));
    void readOrigins().then((answer) => setOrigins(loaded(answer)));
    void readCatalog().then(acceptCatalog);
  }, [acceptCatalog]);

  // Read on opening the view, and again on each return to it: another client
  // may have switched, or a download landed, since.
  useEffect(() => {
    if (visible) reread();
  }, [visible, reread]);

  // The model serves again after a switch, a reload or a failure: what it runs with is read again.
  const lastModel = useRef(model);
  useEffect(() => {
    const before = lastModel.current;
    lastModel.current = model;
    if (model.state === "ready" && (loadUnderWay(before) || before.state === "error")) reread();
  }, [model, reread]);

  // Another model now serves: a draft made against the old one's values is
  // dropped, not applied to the new one. A reload keeps the id, and its draft.
  const readyId = model.state === "ready" ? model.id : null;
  const lastReadyId = useRef(readyId);
  useEffect(() => {
    if (readyId === null) return;
    if (lastReadyId.current !== null && lastReadyId.current !== readyId) {
      setDraft({});
      setMarked([]);
    }
    lastReadyId.current = readyId;
  }, [readyId]);

  useEffect(() => {
    if (!awaiting) return;
    // An unloaded model is followed too: Load on it answers 202 before it loads.
    if (!canSend(model) || !sameSwitch(model.lastSwitch, awaiting.before)) {
      setAwaiting(null);
      return;
    }
    const timer = setInterval(refreshModel, 1000);
    return () => clearInterval(timer);
  }, [awaiting, model, refreshModel]);

  // A download runs: the catalog each second while the view is shown, a failed
  // read keeping the rows last read. Once none runs, the model it fetched is
  // servable: GET /v1/models is read again for the rest of the page.
  useEffect(() => {
    if (!following || !visible) return;
    return followDownloads(readCatalog, (answer) => {
      if (!answer.ok) return;
      acceptCatalog(answer);
      if (!runningDownload(answer.value)) refreshModel();
    });
  }, [following, visible, acceptCatalog, refreshModel]);

  // A reload ends: the draft clears once it took, and is kept when it did not.
  useEffect(() => {
    if (!pending) return;
    const settled = settleReload(pending, model);
    if (settled.kind === "wait") {
      if (settled.pending !== pending) setPending(settled.pending);
      return;
    }
    setPending(null);
    setResult(settled.lines);
    if (settled.ok) {
      setDraft({});
      setMarked([]);
    }
  }, [pending, model]);

  const fieldList = fields.state === "ok" ? fields.value : null;

  const actions = useMemo<ServerActions>(
    () => ({
      load(id) {
        setLoadRefusal(null);
        setDownloadRefusal(null);
        setLoadingId(id);
        void switchTo(id).then((answer) => {
          const after = afterSwitch(answer);
          setLoadingId(null);
          setLoadRefusal(after.refusal);
          if (after.follow) setAwaiting({ before: lastSwitchOf(modelNow.current) });
          if (after.refreshModel) refreshModel();
        });
      },
      download(id) {
        setDownloadRefusal(null);
        setLoadRefusal(null);
        setDownloadingId(id);
        void startDownload(id).then((answer) => {
          const after = afterDownload(answer);
          setDownloadingId(null);
          setDownloadRefusal(after.refusal);
          // The row reads `running` at once: read it now, then each second.
          if (after.follow) void readCatalog().then(acceptCatalog);
        });
      },
      cancel() {
        setDownloadRefusal(null);
        setLoadRefusal(null);
        setCancelling(true);
        void cancelDownload().then((answer) => {
          setCancelling(false);
          if (!answer.ok) setDownloadRefusal(answer.message);
          // The server answers once it stopped: the row reads partial, with Resume, at once.
          void readCatalog().then(acceptCatalog);
        });
      },
      askRemoval(id) {
        setRemoval(null);
        setConfirmingRemoval(id);
      },
      confirmRemoval(id) {
        setConfirmingRemoval(null);
        setRemoval(null);
        setRemovingId(id);
        void removeModel(id).then((answer) => {
          setRemovingId(null);
          setRemoval(afterRemoval(answer));
          // The server answers once the files are gone: the row reads as it now is,
          // and GET /v1/models no longer lists the entry.
          void readCatalog().then(acceptCatalog);
          if (answer.ok) refreshModel();
        });
      },
      keepModel() {
        setConfirmingRemoval(null);
      },
      edit(decl, inForce, value) {
        setDraft((current) => editDraft(current, decl, inForce, value));
        setMarked((current) => current.filter((key) => key !== decl.file_key));
      },
      remove(decl, scope) {
        setDraft((current) => removeDraft(current, decl, scope));
        setMarked((current) => current.filter((key) => key !== decl.file_key));
      },
      undo(decl, scope) {
        setDraft((current) => dropDraft(current, decl, scope));
      },
      discard() {
        setDraft({});
        setMarked([]);
        setResult(null);
      },
      apply() {
        const body = patchBody(draft);
        if (Object.keys(body).length === 0) return;
        setApplying(true);
        setResult(null);
        setMarked([]);
        void patchConfig(body).then((answer) => {
          const before = lastSwitchOf(modelNow.current);
          const after = afterPatch(answer, fieldList, before);
          setApplying(false);
          setResult(after.result);
          setMarked(after.marked);
          if (after.clearDraft) setDraft({});
          if (after.config) {
            setConfig({ state: "ok", value: after.config });
            void readOrigins().then((read) => setOrigins(loaded(read)));
          } else if (after.clearDraft) reread();
          if (after.pending) {
            // The whole patch goes through a reload; the band shows it as a switch.
            setPending(after.pending);
            setAwaiting({ before });
          }
          if (after.refreshModel) refreshModel();
        });
      },
      dismiss() {
        setResult(null);
      },
    }),
    [draft, fieldList, refreshModel, reread, acceptCatalog],
  );

  return [
    { fields, config, origins, catalog, disk, draft, marked, result, pending, applying, loadingId, loadRefusal, downloadingId, downloadRefusal, downloadSpeed: speedOf(progress), cancelling, confirmingRemoval, removingId, removal },
    actions,
  ];
}

const lastSwitchOf = (model: ModelState) => (model.state === "reading" ? undefined : model.lastSwitch);

/** `Date.now()`, ticking each second while `running` — the switch's elapsed seconds — and still otherwise. */
export function useTick(running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [running]);
  return now;
}
