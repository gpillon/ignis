"""#469: an artifact the converter built from the base checkpoint, against the served one, in three tiers.

  python convert.py compare --served SERVED.ninfer --artifact BUILT.ninfer --checkpoint DIR --report R.json
      [--device cpu]

The tiers come from the two recipes' routes (`recipe.served_recipes`, `recipe.recipes`), object by object;
each artifact's sidecar names its recipe (`recipe_id`):

1. **binding**: every object neither calibrated nor taken from the third-party quant (831): the frontend
   resources, the norms and the GDN small set, the BF16 exceptions, the embedding, the heads, the MTP layer,
   the vision tower and the 66 DFlash2 objects. Both recipes make them alike: they must be byte-equal, and a
   difference names the object and fails the comparison (exit 3).
2. the 135 parents both recipes encode locally, and their 135 divisors: expected byte-equal (the encoder is
   deterministic). A parent that differs is measured: its differing words (E2M1 codes, E4M3 scales, the FP32
   weight divisor), its weight divisor's relative difference, and the relative reconstruction error against
   the BF16 rows on both sides; a divisor that differs, its relative difference (`(scratch - served) /
   served`).
3. the 112 parents the served recipe took from the third-party quant (the MLP of layers 0-55), and their
   112 divisors: expected to differ; each parent's reconstruction error is measured on both sides.

It writes the JSON report and, beside it as `.txt`, the summary it prints. The checkpoint is held to the
built artifact's record (its revision and every shard's LFS oid, from the download records: `run` and
`verify` hashed the shards); its BF16 rows are read one shard at a time. Nothing here writes an artifact.

`rebuild` makes objects of the sources with the converter's own `produce` and checks them against an
artifact's, which is how the machine-local test binds tier 1 of the served artifact to the base checkpoint.
"""
import os
import struct
import time

import numpy as np

import checkpoint
import container
import convert
import encode
import inventory
import recipe
import shards
import work
from checkpoint import Refused

EXIT_TIER1_FAILED = 3
BLOCK = 64 << 20
SIDECARS = (".conversion.json", ".graft.json")
# recipe_id -> its routes
RECIPES = {convert.RECIPE_ID: recipe.recipes, recipe.SERVED_RECIPE_ID: recipe.served_recipes}
TIER1_ROUTES = ("resource", "base_direct", "bf16_exception", "official", "dflash2")
# (the served recipe's route, the local recipe's) -> tier
TIER_OF = {**{(r, r): 1 for r in TIER1_ROUTES},
           ("local_nvfp4", "local_nvfp4"): 2, ("input_divisor", "input_divisor"): 2,
           ("source_nvfp4", "local_nvfp4"): 3, ("source_input_divisor", "input_divisor"): 3}


def tiers(served, local):
    """object name -> tier (1, 2 or 3) of every object of the two recipes (name -> Recipe); refused, naming
    them, when an object's routes put it in no tier."""
    out, problems = {}, []
    for name in sorted(set(served) | set(local)):
        routes = (getattr(served.get(name), "route", None), getattr(local.get(name), "route", None))
        if routes in TIER_OF:
            out[name] = TIER_OF[routes]
        else:
            problems.append(f"{name} is routed {routes[0]} by the served recipe and {routes[1]} by the built one")
    checkpoint.refuse_if(problems, "the recipes put objects in no tier")
    return out


def tier_names(g, tier):
    """The names of a tier's objects at geometry `g`, in the inventory's order."""
    of = tiers(recipe.served_recipes(g), recipe.recipes(g))
    return [o.name for o in inventory.objects(g) if of[o.name] == tier]


# ---------------------------------------------------------------------------------------------------
# Bytes
# ---------------------------------------------------------------------------------------------------

def differing_bytes(a, b):
    """How many bytes of `a` and `b` differ, a length difference counting as differing bytes."""
    n = min(len(a), len(b))
    return int(np.count_nonzero(np.frombuffer(a, np.uint8, n) != np.frombuffer(b, np.uint8, n))) + abs(len(a) - len(b))


def stored_blocks(found, name, block=BLOCK):
    """`name`'s payload in `found` (a `container.Container`), `block` bytes at a time."""
    offset, size = found.span(name)
    with open(found.path, "rb") as f:
        f.seek(offset)
        while size > 0:
            data = f.read(min(block, size))
            if not data:
                raise ValueError(f"{found.path}: {name} extends beyond the file")
            size -= len(data)
            yield data


def differing_from(found, name, payload):
    """How many bytes of `payload` differ from `name`'s in `found`, read a block at a time."""
    view = memoryview(payload)
    at = differ = 0
    for data in stored_blocks(found, name):
        m = max(0, min(len(data), len(view) - at))
        differ += differing_bytes(data[:m], view[at: at + m])
        at += len(data)
    return differ + abs(at - len(payload))


def rebuild(found, sources, table, ids, wanted, device):
    """The objects `wanted` that `table`'s routes make of `sources` (the converter's `produce`), each against
    `found`'s: {objects, bytes, differ: [{name, route, bytes, differing_bytes}]}."""
    out = {"objects": 0, "bytes": 0, "differ": []}
    for o, payload, _ in convert.produce(sources, table, {}, ids, device, wanted):
        differ = differing_from(found, o.name, payload)
        out["objects"] += 1
        out["bytes"] += len(payload)
        if differ:
            out["differ"].append({"name": o.name, "route": table[o.name].route, "bytes": len(payload),
                                  "differing_bytes": differ})
    return out


# ---------------------------------------------------------------------------------------------------
# The two artifacts and the checkpoint
# ---------------------------------------------------------------------------------------------------

class Side:
    """One artifact of the comparison: its container, its sidecar (`.conversion.json` or `.graft.json`) and
    the routes of the recipe the sidecar names; refused unless the three agree."""

    def __init__(self, path, g, what):
        self.path = path
        self.found = container.Container(path)
        sidecar = next((path + s for s in SIDECARS if os.path.isfile(path + s)), None)
        if sidecar is None:
            raise Refused(f"{what} {path} has no {' or '.join(SIDECARS)} beside it")
        self.sidecar_path = sidecar
        self.sidecar = checkpoint.read_json(sidecar, f"{what}'s sidecar {sidecar}")
        self.recipe_id = self.sidecar.get("recipe_id")
        if self.recipe_id not in RECIPES:
            raise Refused(f"{what}'s sidecar {sidecar} names recipe {self.recipe_id}, which compare does not know "
                          f"(it knows {', '.join(sorted(RECIPES))})")
        self.table = RECIPES[self.recipe_id](g)
        problems = convert.directory_problems(self.found, g, self.found.identity.get("weights_id"))
        size, recorded = os.path.getsize(path), (self.sidecar.get("artifact") or {}).get("bytes")
        if recorded != size:
            problems.append(f"it is {size} bytes, its sidecar records {recorded} bytes")
        count = (self.sidecar.get("objects") or {}).get("count")
        if count != len(self.found.objects):
            problems.append(f"it has {len(self.found.objects)} objects, its sidecar records {count}")
        local = {n for n, r in self.table.items() if r.route in ("local_nvfp4", "dflash2")}
        problems += [f"its sidecar records {p} as encoded locally, which recipe {self.recipe_id} does not"
                     for p in sorted(((self.sidecar.get("local_nvfp4") or {}).get("parents") or {}))
                     if p not in local]
        checkpoint.refuse_if(problems, f"{what} {path} is not what its sidecar records")

    def described(self):
        return {"artifact": os.path.basename(self.path), "bytes": os.path.getsize(self.path),
                "identity": self.found.identity, "sidecar": os.path.basename(self.sidecar_path),
                "recipe_id": self.recipe_id}


def bound_checkpoint(g, root, ours):
    """The checkpoint's sources through the cheap checks, refused unless they are the ones `ours` records it
    was converted from: the revision and every shard's LFS oid, from the download records."""
    sources = checkpoint.check_sources(g, root)
    source = ours.sidecar.get("source") or {}
    files = {n: (p or {}).get("sha256") for n, p in (source.get("files") or {}).items() if n.endswith(".safetensors")}
    differ = sorted(n for n in set(files) | set(sources.shard_oids) if files.get(n) != sources.shard_oids.get(n))
    if source.get("revision") != sources.revision or differ:
        what = differ[0] if differ else f"revision {sources.revision}"
        raise Refused(f"the checkpoint {root} is not the one the artifact was converted from: its {what} is not "
                      f"the one {os.path.basename(ours.sidecar_path)} records (revision {source.get('revision')})")
    return sources


# ---------------------------------------------------------------------------------------------------
# The tiers
# ---------------------------------------------------------------------------------------------------

def tier1(served, ours, names):
    out = {"objects": 0, "bytes": 0, "byte_equal": 0, "differ": []}
    for name in names:
        differ = sum(differing_bytes(a, b) for a, b in zip(stored_blocks(served.found, name),
                                                             stored_blocks(ours.found, name)))
        size = served.found.span(name)[1]
        out["objects"] += 1
        out["bytes"] += size
        if differ:
            out["differ"].append({"name": name, "route": ours.table[name].route, "bytes": size,
                                  "differing_bytes": differ})
        else:
            out["byte_equal"] += 1
    return out


def pair(served, scratch):
    return {"served": served, "scratch": scratch,
            "relative_difference": None if served == 0.0 else (scratch - served) / served}


def divisor_entry(served, ours, name):
    s, o = served.found.read(name), ours.found.read(name)
    return dict({"name": name, "byte_equal": s == o}, **pair(struct.unpack("<f", s)[0], struct.unpack("<f", o)[0]))


def parent_entry(served, ours, o, rows, device, measure):
    """A parent's comparison: byte-equal, else (and always when `measure`) its differing words, its weight
    divisors and both sides' reconstruction error against its BF16 rows (`rows()`); its divisor's."""
    s, b = served.found.read(o.name), ours.found.read(o.name)
    entry = {"name": o.name, "byte_equal": s == b}
    if measure or not entry["byte_equal"]:
        n, k = o.shape
        scale_offset = inventory.block_scale_geometry(o.shape)[0]
        sp, ss, sd = encode.nvfp4_unlayout(s, o.shape)
        bp, bs, bd = encode.nvfp4_unlayout(b, o.shape)
        codes = (sp ^ bp).numpy()
        entry["differing_words"] = {
            "codes": int(np.count_nonzero(codes & 0x0F)) + int(np.count_nonzero(codes & 0xF0)), "codes_of": n * k,
            "scales": int(np.count_nonzero((ss != bs).numpy())), "scales_of": n * k // 16,
            "weight_divisor": int(s[-4:] != b[-4:]),
            "padding_bytes": differing_bytes(s[n * k // 2: scale_offset], b[n * k // 2: scale_offset])}
        entry["weight_divisor"] = pair(sd, bd)
        weight = rows()
        entry["relative_frobenius_error"] = {"served": encode.nvfp4_error(weight, sp, ss, sd, device),
                                             "scratch": encode.nvfp4_error(weight, bp, bs, bd, device)}
    entry["input_divisor"] = divisor_entry(served, ours, inventory.divisor_name(o.name))
    return entry


def parents(served, ours, objs, reader, device, measure, tier):
    entries = []
    for i, o in enumerate(objs, 1):
        r = ours.table[o.name]
        e = parent_entry(served, ours, o, lambda: recipe.materialize(r, o, reader), device, measure)
        entries.append(e)
        if "relative_frobenius_error" in e:
            err = e["relative_frobenius_error"]
            convert.log(f"tier {tier} {i}/{len(objs)}: {o.name}: error served {err['served']:.6f}, "
                        f"scratch {err['scratch']:.6f}")
    return entries


def stats(entries, side):
    errors = [e["relative_frobenius_error"][side] for e in entries]
    return {"mean": sum(errors) / len(errors), "max": max(errors)} if errors else None


# ---------------------------------------------------------------------------------------------------
# The summary and the command
# ---------------------------------------------------------------------------------------------------

def rel(value):
    return "-" if value is None else f"{value:+.3e}"


def summary(report):
    s, a = report["served"], report["artifact"]
    lines = [f"compare (#469): {a['artifact']} ({a['identity'].get('weights_id')}, recipe {a['recipe_id']}) "
             f"against the served {s['artifact']} ({s['identity'].get('weights_id')}, recipe {s['recipe_id']})"]
    t1 = report["tier1"]
    lines.append(f"tier 1, binding: {t1['objects']} objects, {t1['byte_equal']} byte-equal ({t1['bytes']} bytes)")
    lines += [f"  differs: {d['name']} ({d['route']}): {d['differing_bytes']} of {d['bytes']} bytes"
              for d in t1["differ"]]
    t2 = report["tier2"]
    lines.append(f"tier 2: {t2['parents']} parents both encode locally, {t2['parents_byte_equal']} byte-equal; "
                 f"their divisors, {t2['divisors_byte_equal']} byte-equal")
    for e in t2["entries"]:
        if not e["byte_equal"]:
            w, err = e["differing_words"], e["relative_frobenius_error"]
            lines.append(f"  {e['name']}: {w['codes']} of {w['codes_of']} codes, {w['scales']} of {w['scales_of']} "
                         f"scales and {w['padding_bytes']} padding bytes differ; weight divisor served "
                         f"{e['weight_divisor']['served']!r}, scratch {e['weight_divisor']['scratch']!r} "
                         f"({rel(e['weight_divisor']['relative_difference'])}); error against the BF16 served "
                         f"{err['served']:.6f}, scratch {err['scratch']:.6f}")
        d = e["input_divisor"]
        if not d["byte_equal"]:
            lines.append(f"  {d['name']}: served {d['served']!r}, scratch {d['scratch']!r} "
                         f"({rel(d['relative_difference'])})")
    t3 = report["tier3"]
    if t3["entries"]:
        lines.append(f"tier 3: {t3['parents']} parents the served artifact took from the third-party quant; relative "
                     f"reconstruction error against the BF16: served mean {t3['served_error']['mean']:.6f}, max "
                     f"{t3['served_error']['max']:.6f}; scratch mean {t3['scratch_error']['mean']:.6f}, max "
                     f"{t3['scratch_error']['max']:.6f}; scratch below served on {t3['scratch_below_served']} of "
                     f"{t3['parents']}; divisors byte-equal {t3['divisors_byte_equal']}")
        lines.append(f"  {'parent':<30} {'served':>9} {'scratch':>9} {'d_x served':>13} {'d_x scratch':>13} "
                     f"{'d_x rel':>10}")
        for e in t3["entries"]:
            err, d = e["relative_frobenius_error"], e["input_divisor"]
            lines.append(f"  {e['name']:<30} {err['served']:>9.6f} {err['scratch']:>9.6f} {d['served']:>13.6g} "
                         f"{d['scratch']:>13.6g} {rel(d['relative_difference']):>10}")
    lines.append(report["verdict"])
    return "\n".join(lines) + "\n"


def run(args, g, device):
    """The `compare` command: refusals (exit 2) come before anything is read past the sidecars and headers."""
    served = Side(args.served, g, "the served artifact")
    ours = Side(args.artifact, g, "the artifact")
    tier = tiers(served.table, ours.table)
    sources = bound_checkpoint(g, args.checkpoint, ours)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    t0 = time.time()
    objs = inventory.objects(g)
    report = {"served": served.described(), "artifact": ours.described(),
              "checkpoint": {"revision": sources.revision}, "device": str(device)}
    report["tier1"] = tier1(served, ours, [o.name for o in objs if tier[o.name] == 1])
    convert.log(f"tier 1: {report['tier1']['byte_equal']} of {report['tier1']['objects']} objects byte-equal")
    nvfp4 = [o for o in objs if o.format == inventory.NVFP4 and o.unit == "text"]
    with shards.ShardReader(args.checkpoint, sources.weight_map) as reader:
        second = parents(served, ours, [o for o in nvfp4 if tier[o.name] == 2], reader, device, False, 2)
        third = parents(served, ours, [o for o in nvfp4 if tier[o.name] == 3], reader, device, True, 3)
    report["tier2"] = {"parents": len(second), "parents_byte_equal": sum(e["byte_equal"] for e in second),
                       "divisors_byte_equal": sum(e["input_divisor"]["byte_equal"] for e in second),
                       "entries": second}
    report["tier3"] = {"parents": len(third), "parents_byte_equal": sum(e["byte_equal"] for e in third),
                       "divisors_byte_equal": sum(e["input_divisor"]["byte_equal"] for e in third),
                       "served_error": stats(third, "served"), "scratch_error": stats(third, "scratch"),
                       "scratch_below_served": sum(e["relative_frobenius_error"]["scratch"]
                                                   < e["relative_frobenius_error"]["served"] for e in third),
                       "entries": third}
    report["verdict"] = "TIER 1 FAILED" if report["tier1"]["differ"] else "TIER 1 PASS"
    report["seconds"] = round(time.time() - t0, 1)
    report["peaks"] = convert.peaks(device)
    work.write_json_atomic(args.report, report)
    text = summary(report)
    with open(os.path.splitext(args.report)[0] + ".txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    convert.log_step(f"compare: {len(objs)} objects", time.time() - t0, report["peaks"])
    convert.log(text.rstrip("\n"))
    return EXIT_TIER1_FAILED if report["tier1"]["differ"] else 0
