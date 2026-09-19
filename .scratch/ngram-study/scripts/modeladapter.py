"""One shape for two models.

The code is tuned on a small model and then run on Qwen3.8-27B
(`02-storage.md` §8), and the two do not agree on where the decoder layers
live: `model.model.layers` for Qwen2/Qwen3, `model.model.language_model.layers`
for Qwen3_5, which wraps a vision tower alongside the text stack.  Every
accessor the rest of the study needs goes through here, so the extraction and
injection code is byte-identical across the two runs and the small model is a
real rehearsal rather than a different experiment.
"""

import os

# Files in this corpus differ in length, so every forward asks the caching
# allocator for a differently shaped block.  With the default segments the
# reserved pool drifts upward from a 20.6 GiB working set to 31.9 GiB of the
# card's 32.6, at which point WDDM starts paging and a 2.1 s forward becomes
# tens of seconds — the same failure mode this repo already met in ADR 0030.
# Expandable segments keep the reservation near the working set.  This has to
# be set before CUDA initialises, so it lives at import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# unsloth's NVFP4 recipe leaves these unquantized on purpose: they are the
# Gated DeltaNet gating and decay projections, where 4 bits perturb the
# recurrent state far more than they do in a 5120x17408 MLP.  Quantizing them
# would put a confound between "no signal" and "a broken recurrence", so the
# 4-bit config mirrors that choice.  `lm_head` is excluded because the NLL is
# read off it.
SKIP_4BIT = ["lm_head", "visual", "in_proj_a", "in_proj_b"]


def decoder_layers(model):
    """The list of decoder layers, innermost text stack first."""
    base = getattr(model, "model", model)
    for attr in ("language_model", "text_model"):
        inner = getattr(base, attr, None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers
    if hasattr(base, "layers"):
        return base.layers
    raise AttributeError("no decoder layer list on %s" % type(model).__name__)


def num_layers(model):
    return len(decoder_layers(model))


def hidden_size(model):
    cfg = model.config
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return text.hidden_size


def lm_head(model):
    head = getattr(model, "lm_head", None)
    if head is None:
        raise AttributeError("no lm_head on %s" % type(model).__name__)
    return head


def final_norm(model):
    """The norm applied after the last decoder layer, before `lm_head`."""
    base = getattr(model, "model", model)
    for attr in ("language_model", "text_model"):
        inner = getattr(base, attr, None)
        if inner is not None and hasattr(inner, "norm"):
            return inner.norm
    return getattr(base, "norm", None)


def layer_output(out):
    """Decoder layers return either a tensor or a tuple whose first element is
    the hidden state; 5.x is inconsistent about which."""
    return out[0] if isinstance(out, tuple) else out


def rewrap_layer_output(out, new_hidden):
    if isinstance(out, tuple):
        return (new_hidden,) + tuple(out[1:])
    return new_hidden


def set_deterministic(seed=0):
    """The alpha=0 control in `02-storage.md` §7 Fase 3 is only readable if two
    identical runs give identical numbers; this is what makes that true, and
    `phase0.py --determinism` is what checks it."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_model(path, dtype=torch.bfloat16, device_map="cuda", quant="nf4",
               attn_implementation="sdpa"):
    """Load a causal LM for measurement: eval mode, no grad, 4-bit unless asked
    otherwise.  Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"dtype": dtype, "device_map": device_map,
              "attn_implementation": attn_implementation}
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=SKIP_4BIT,
        )
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    model.requires_grad_(False)
    return model, tok
