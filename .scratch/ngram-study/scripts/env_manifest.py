"""Record the identity of everything a result depends on.

`02-storage.md` §10 wants the exact parameters and the model identity next to
the numbers.  A result whose model revision is not recorded cannot be compared
against a later one, so this runs before the first measurement and again
whenever the vehicle changes.
"""

import argparse
import json
import subprocess
import sys
import urllib.request


def _hub_revision(repo):
    try:
        with urllib.request.urlopen(
                "https://huggingface.co/api/models/%s" % repo, timeout=30) as r:
            return json.loads(r.read().decode()).get("sha")
    except Exception as exc:                     # offline is not fatal
        return "unavailable: %s" % exc


def _git(root, *args):
    try:
        return subprocess.check_output(("git", "-C", root) + args,
                                       text=True).strip()
    except Exception as exc:
        return "unavailable: %s" % exc


def build(root, model_repo, model_path, pinned_revision):
    import torch
    import transformers

    versions = {"python": sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": transformers.__version__}
    for name in ("compressed_tensors", "bitsandbytes", "accelerate",
                 "safetensors", "huggingface_hub"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None

    dev = {}
    if torch.cuda.is_available():
        dev = {"name": torch.cuda.get_device_name(0),
               "capability": list(torch.cuda.get_device_capability(0)),
               "total_bytes": torch.cuda.get_device_properties(0).total_memory,
               "arch_list": torch.cuda.get_arch_list()}

    live = _hub_revision(model_repo)
    return {
        "versions": versions,
        "cuda_available": torch.cuda.is_available(),
        "device": dev,
        "model": {
            "repo": model_repo,
            "local_path": model_path,
            "pinned_revision": pinned_revision,
            "hub_main_revision": live,
            "matches_pin": live == pinned_revision,
        },
        "repo": {
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git(root, "rev-parse", "HEAD"),
            "dirty": bool(_git(root, "status", "--porcelain")),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model-repo", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--pinned-revision",
                    default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                    help="the revision the ignis artifact manifest pins")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = build(args.root, args.model_repo, args.model_path,
                args.pinned_revision)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(env, fh, indent=2, sort_keys=True)
    print(json.dumps(env, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
