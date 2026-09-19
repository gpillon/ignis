"""Quantize the BF16 checkpoint to 4-bit once, and keep the result locally.

The base checkpoint is 51.8 GiB and fits on none of this box's local drives, so
it lives on a network share that reads at about 40 MiB/s.  bitsandbytes
quantizes shard by shard as the weights stream in, so this reads the share
exactly once, and everything afterwards loads the ~16 GiB result from a local
disk.

Quantizing is also what makes the model fit at all: transformers 5.17 hands
every compressed-tensors checkpoint that is not pure FP8 to a hook that
decompresses it to dense BF16 on the first forward, so the published NVFP4
copy would need 54 GB of VRAM rather than 21.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeladapter import SKIP_4BIT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    dtype = getattr(torch, args.dtype)
    cfg = AutoConfig.from_pretrained(args.src)
    print("architectures:", getattr(cfg, "architectures", None), flush=True)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=SKIP_4BIT,
    )

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.src, dtype=dtype, device_map="cuda",
        quantization_config=quant, attn_implementation="sdpa")
    load_s = time.time() - t0
    print("loaded in %.1f min" % (load_s / 60), flush=True)
    print("cuda allocated %.2f GiB, peak %.2f GiB"
          % (torch.cuda.memory_allocated() / 2 ** 30,
             torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)

    os.makedirs(args.dst, exist_ok=True)
    t1 = time.time()
    model.save_pretrained(args.dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.src).save_pretrained(args.dst)
    save_s = time.time() - t1

    size = sum(os.path.getsize(os.path.join(args.dst, f))
               for f in os.listdir(args.dst)
               if os.path.isfile(os.path.join(args.dst, f)))
    report = {
        "src": args.src, "dst": args.dst,
        "load_seconds": round(load_s, 1), "save_seconds": round(save_s, 1),
        "cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
        "dst_bytes": size, "dst_gib": round(size / 2 ** 30, 2),
        "skip_modules": SKIP_4BIT,
        "quant": "nf4/double/bf16-compute",
    }
    print(json.dumps(report, indent=2), flush=True)
    with open(os.path.join(args.dst, "convert_report.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
