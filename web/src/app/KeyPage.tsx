import { type FormEvent, useState } from "react";
import { checkKey, saveKey } from "../api/auth.ts";
import flame from "../brand/flame.webp";
import wordmark from "../brand/wordmark-light.webp";
import { caption, field } from "../ui/classes.ts";

/** The page ignis's 401 leads to: asks for the API key, checks it, keeps it in this browser. */
export function KeyPage({ rejected }: { rejected: boolean }) {
  const [key, setKey] = useState("");
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(rejected ? "ignis refused the saved key." : null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const candidate = key.trim();
    if (!candidate || checking) return;
    setChecking(true);
    const result = await checkKey(candidate);
    setChecking(false);
    if (result.ok) saveKey(candidate);
    else setError(result.message);
  }

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <header className="z-20 shrink-0 bg-kiln text-[#eae8e4]">
        <div className="flex items-center gap-3 px-4 py-3 sm:gap-4 md:px-6">
          <img src={flame} alt="" className="flame h-9 w-auto" />
          <img src={wordmark} alt="ignis" className="h-[18px] w-auto" />
          <span className="hidden h-6 w-px bg-kiln-line sm:block" aria-hidden />
          <span className="hidden font-display text-[15px] font-medium tracking-wide text-[#b9bec4] sm:block">Playground</span>
        </div>
        <div className="heat" aria-hidden />
      </header>

      <main className="m-auto flex w-full max-w-sm flex-col items-center gap-7 px-4 py-10 text-center">
        <img src={flame} alt="" className="h-32 w-auto drop-shadow-[0_18px_40px_rgb(200_22_29/0.35)]" />
        <div className="flex flex-col gap-2">
          <h1 className="font-display text-4xl font-semibold tracking-tight text-ink">API key required</h1>
          <p className="text-sm text-ash">
            This ignis runs with <span className="font-mono text-ink">--api-key</span>. Enter the key to open the
            Playground; it stays in this browser.
          </p>
        </div>
        <form onSubmit={submit} className="flex w-full flex-col gap-2 text-left">
          <label htmlFor="api-key" className={caption}>
            API key
          </label>
          <input
            id="api-key"
            name="api-key"
            type="password"
            autoComplete="current-password"
            autoFocus
            spellCheck={false}
            value={key}
            onChange={(e) => {
              setKey(e.target.value);
              setError(null);
            }}
            className={`${field} font-mono`}
          />
          {error && (
            <p role="alert" className="text-sm text-fault">
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={!key.trim() || checking}
            className="cut mt-2 bg-ember px-4 py-2.5 font-display text-sm font-semibold text-white [--cut-size:8px] hover:brightness-110 disabled:opacity-50"
          >
            {checking ? "Checking…" : "Unlock"}
          </button>
        </form>
      </main>
    </div>
  );
}
