"""The calibration: every activation divisor `d_x` of the 247 text NVFP4 parents, measured on the BF16
checkpoint over a fixed corpus (a port of ninfer's `calibrate_nvfp4full.py`, local-only).

The checkpoint does not fit on the card, so the model is built with empty weights and each decoder layer's
parameters are read in before its forward and dropped after it; only the token embedding and the final
norm stay. Every site records the largest absolute BF16 input it sees over the corpus, one concatenated
prefill of the ten documents below:

    d_x = binary32(2688 / max|site input|)        # 2688 = 6 * 448

the A4 block scale's orientation, `s_x = E4M3FN(d_x * max|block| / 6)`. The sites are the parents' inputs:
`self_attn.q_proj` (the fused query, key, gate and value), `self_attn.o_proj`, `linear_attn.in_proj_qkv`
(the fused query, key, value and z), `linear_attn.out_proj`, `mlp.gate_proj` (gate and up) and
`mlp.down_proj`. ninfer measured 135 of them and copied the other 112 from a third-party quant; here all
247 are measured.

Two changes from ninfer's: the parameters are read through `shards.ShardReader`, one shard open at a time
(ninfer's loader keeps a handle on every shard it reads, 55.6 GB of commit charge on Windows); and the GDN's
chunked delta rule is transformers' reference torch function unless `--gdn-kernel installed` asks for the
kernel transformers would pick (flash-linear-attention's, when it is installed, which runs on CUDA only).
"""
import hashlib
import inspect
import sys
import time

import torch

import encode
import inventory
import shards

# The fixed calibration corpus, ninfer's `_CALIBRATION_DOCUMENTS` verbatim: prose, technical writing,
# source code and numeric tables, in four languages.
DOCUMENTS = (
    """The history of numerical weather prediction begins with Lewis Fry Richardson's 1922
attempt to compute atmospheric pressure changes by hand. His forecast, produced by a room
of human calculators working in shifts, took six weeks to advance the atmosphere by six
hours and was wildly inaccurate. Yet the method he described - dividing the atmosphere into
a grid of cells and applying the equations of fluid dynamics to each cell - is precisely
what every modern forecast model does. The difference is that where Richardson needed
weeks of human labor, a contemporary supercomputer performs the same arithmetic in a few
seconds, on a grid a hundred times finer, and assimilates observations from satellites,
aircraft, ocean buoys, and ground stations to initialize the computation. The forecasting
problem is now less about raw computation than about representing small-scale physics:
clouds, turbulence, and convection that fall below the grid resolution.""",
    """Gradient-based optimization underlies nearly all of modern machine learning. Given a
differentiable loss function L parameterized by weights w, training iterates w <- w - eta
* grad L(w), where eta is the learning rate. Stochastic gradient descent estimates the
gradient from minibatches; momentum accumulates a velocity v <- mu * v - eta * grad to
damp oscillations across ill-conditioned valleys; and adaptive methods such as Adam
maintain per-parameter running estimates of the first and second moments, rescaling the
step by an estimate of the gradient's magnitude. Careful initialization matters: drawing
weights from a distribution whose variance matches the fan-in and fan-out of each layer
keeps activation magnitudes stable as signals propagate forward and gradients propagate
backward. Normalization layers - batch norm, layer norm, RMSNorm - remove mean and scale
drift, allowing higher learning rates. Learning-rate schedules decay eta over training:
warmup ramps it up over the first few thousand steps, cosine schedules anneal it toward
zero, and step schedules divide it by a constant at fixed milestones. Regularization by
weight decay, dropout, and data augmentation counteracts overfitting.""",
    """def build_frequency_table(samples: list[int], size: int) -> list[float]:
    counts = [0] * size
    for value in samples:
        if not 0 <= value < size:
            raise ValueError(f"sample out of range: {value}")
        counts[value] += 1
    total = sum(counts)
    if total == 0:
        return [0.0] * size
    return [count / total for count in counts]

def entropy(table: list[float]) -> float:
    from math import log2
    return -sum(p * log2(p) for p in table if p > 0.0)

def rolling_checksum(data: bytes, window: int = 32, modulus: int = 16777619) -> int:
    if window <= 0:
        raise ValueError("window must be positive")
    digest = 0
    for i, byte in enumerate(data):
        digest = (digest * 31 + byte) % modulus
        if i >= window:
            digest = (digest - data[i - window] * pow(31, window, modulus)) % modulus
    return digest

class RingBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: list[float] = []
        self.head = 0

    def push(self, value: float) -> float | None:
        evicted = None
        if len(self.items) == self.capacity:
            evicted = self.items[self.head]
            self.items[self.head] = value
            self.head = (self.head + 1) % self.capacity
        else:
            self.items.append(value)
        return evicted""",
    """Quarterly report extract, all figures in thousands of dollars unless noted.
Revenue: Q1 4,812; Q2 5,377; Q3 5,940; Q4 6,512; full year 22,641, up 18.4 percent
year over year. Cost of revenue: Q1 2,105; Q2 2,318; Q3 2,504; Q4 2,691. Gross margin
improved from 56.2 percent in Q1 to 58.7 percent in Q4. Operating expenses: research and
development 6,204, sales and marketing 4,481, general and administrative 1,902. Operating
income 2,363 for the year, margin 10.4 percent. Net income 1,977 after interest expense
of 214 and tax provision of 172 at an effective rate of 9.6 percent. Cash and equivalents
11,340 at year end; inventory 2,905; accounts receivable 3,661 with days sales
outstanding of 51. Capital expenditure 1,208, of which 744 related to compute capacity.
Headcount ended at 612, up from 545, of which 289 in engineering. Contracted backlog
stood at 9,880 with average duration of 2.3 years.""",
    """Le vent se levait sur la plaine de Beauce balayant les champs de blé presque mûrs.
Les paysans avaient fini la moisson dans le voisinage et les meules dorées attendaient
les chariots. Au loin, on apercevait la flèche de la cathédrale, grise sur le ciel
encore clair, tandis que des nuages épais montaient à l'horizon du côté de la forêt.
La chaleur de la journée s'attardait dans l'air immobile du soir. On entendait
seulement le froissement des feuilles sèches et, de temps en temps, l'aboiement d'un
chien dans quelque ferme éloignée. Elle marcha longtemps sur le chemin creux entre les
haies, ne pensant à rien, regardant la terre de Beauce s'endormir lentement dans la
lumière déclinante. Il lui semblait que cette plaine interminable portait tous les
travaux et toutes les saisons de sa vie, et que chaque sillon connaissait son nom.""",
    """一九四三年秋，昆明的雨季来得比往年早。联大的教室屋顶是铁皮的，雨点打上去，
声音大得讲课的人都听不见自己说话。教授停下来，学生们便合上笔记，听雨。有人
后来回忆说，那几年的知识有一半是在雨声里学的。物价一日三涨，教授们在实验室
外面种菜，在中学兼课，把藏书一册一册地卖出去。可是图书馆晚上依然坐满了人，
一盏油灯下面常常是两个人。物理系的仪器用完了就拆，拆了又装。跑警报的日子，
师生们在郊外的山沟里继续讨论功课，有人带着论文字典，有人带着未完成的实验
记录。那些年在昆明写成的论文，后来散落在世界各地，但其中许多的初稿，是在
铁皮屋顶下、油灯旁边、山沟里的石板上完成的。""",
    """{
  "session": {"id": "a4f7c2e1", "started": "2026-04-03T09:14:22Z", "region": "eu-west-1"},
  "customer": {"tier": "enterprise", "seats": 1240, "renewal": "2027-01-15"},
  "usage": {
    "inference": {"requests": 8421937, "tokens_in": 1120394221, "tokens_out": 402118336},
    "training": {"jobs": 217, "gpu_hours": 18432.5, "checkpoint_gb": 91.3}
  },
  "incidents": [
    {"id": "INC-2201", "severity": 2, "started": "2026-03-28T21:04:11Z",
     "duration_minutes": 43, "affected": ["inference", "dashboard"],
     "cause": "upstream capacity provider failure", "resolved": true},
    {"id": "INC-2207", "severity": 3, "started": "2026-04-01T02:33:57Z",
     "duration_minutes": 11, "affected": ["batch-ingest"], "resolved": true}
  ],
  "capacity": {"gpu_allocated": 512, "gpu_used_peak": 489, "queue_depth_p99": 74}
}""",
    """El camión arrancó al amanecer y tomó la carretera de tierra que subía hacia la
sierra. En la caja, entre sacos de maíz y bidones de agua, iban once personas y
dos gallos. El conductor, que había hecho ese viaje dos veces por semana durante
veinte años, no necesitaba mirar el camino: sabía el nombre de cada curva, de
cada zanja, de cada árbol caído que había que esquivar. A media mañana pararon
en un paraje donde había una cruz de madera y una pileta de agua clara. Allí
comieron tortillas con frijoles y queso, y el más viejo del grupo contó otra vez
la historia del túnel de la mina, la fiebre del oro del año cuarenta y ocho, y
cómo el río cambió de curso una noche de tormenta y se llevó la mitad del pueblo
viejo. Nadie lo interrumpía, aunque todos se la sabían de memoria.""",
    """In compiler design, register allocation is traditionally formulated as a graph
coloring problem: build an interference graph whose nodes are program variables
(live ranges) and whose edges connect variables live at the same program point,
then color the graph with k colors, where k is the number of available registers.
Chaitin's allocator spills a variable when no color can be found, inserting load
and store instructions around the definition and uses, then rebuilding the graph
because spilling changes live ranges. Linear-scan allocation, by contrast, sorts
live intervals by start position and walks them once, keeping an active list and
evicting the interval with the furthest end point when a register is needed -
a single pass with worse register quality but predictable, fast compilation,
which made it the standard choice for just-in-time compilers. SSA form makes
interference sparser: variables defined once interfere only when their live
ranges overlap, and the dominance ordering of definitions enables coalescing of
copy instructions through parallel-copy elimination and lost-copy avoidance.""",
    """# Migration notes: v4 to v5

The v5 release rewrites the storage engine. Read this before upgrading.

**Breaking changes**

1. The manifest format changed from YAML to a length-prefixed binary envelope.
   Run `ninfer migrate --in place/ --out place2/` once; it is idempotent and
   verifies every page checksum before touching the destination.
2. `GET /v4/objects` is removed. Use `GET /v5/objects?cursor=...`; responses
   now return at most 500 names and include a `next_cursor` field.
3. Timestamps are microseconds since epoch (int64), not ISO strings.

**Deprecations**

- `--sort-key` still works but is ignored; v5 always returns canonical order.
- The `x-retention` header moved to the object metadata envelope.

**Operational checklist**

- [ ] Free disk space >= 2.2x the v4 dataset size
- [ ] Backup of the manifest and at least the two most recent snapshots
- [ ] Drain writers (v5 tolerates one stale writer for at most 300 seconds)
- [ ] After migration: `ninfer verify --deep` (about 40 minutes per terabyte)

Rollback: keep the v4 directory until `verify --deep` passes twice in a row.""",
)
SITE_MODULES = {"query_key_gate_value": "self_attn.q_proj", "output@attention": "self_attn.o_proj",
                "query_key_value_z": "linear_attn.in_proj_qkv", "output@gdn": "linear_attn.out_proj",
                "gate_up": "mlp.gate_proj", "down": "mlp.down_proj"}


def corpus():
    return "\n\n".join(DOCUMENTS)


def corpus_sha256():
    return hashlib.sha256(corpus().encode("utf-8")).hexdigest()


def sites(g):
    """divisor object -> the module whose input it measures, for the 247 text NVFP4 parents."""
    out = {}
    for o in inventory.objects(g):
        if o.unit != "text" or o.format != inventory.NVFP4:
            continue
        _, _, layer, block, suffix = o.name.split("/")
        key = f"{suffix}@{block}" if suffix == "output" else suffix
        out[inventory.divisor_name(o.name)] = f"language_model.layers.{layer}.{SITE_MODULES[key]}"
    return out


def gdn_reference():
    """Make the GDN's chunked delta rule transformers' reference torch function; returns its name.
    flash-linear-attention is kept from importing (it initializes CUDA on import, a CPU run included)."""
    if "transformers.models.qwen3_5.modeling_qwen3_5" not in sys.modules:
        sys.modules.setdefault("fla", None)
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5
    qwen3_5.torch_chunk_gated_delta_rule = inspect.unwrap(qwen3_5.torch_chunk_gated_delta_rule)
    f = qwen3_5.torch_chunk_gated_delta_rule
    return f"{f.__module__}.{f.__qualname__}"


class StreamingLayers:
    """Each decoder layer's parameters read in (onto `device`) before its forward and dropped after it."""

    def __init__(self, model, reader, device):
        self.model, self.reader, self.device = model, reader, device
        self.handles = []

    @staticmethod
    def parameters_of(module, prefix):
        for child, sub in module.named_children():
            yield from StreamingLayers.parameters_of(sub, f"{prefix}.{child}")
        for attr, p in list(module._parameters.items()):
            if p is not None:
                yield module, attr, f"{prefix}.{attr}"

    def install(self, module, attr, path):
        t = self.reader.get("model." + path).to(self.device)
        module._parameters[attr] = torch.nn.Parameter(t, requires_grad=False)

    @staticmethod
    def evict(module, attr):
        p = module._parameters[attr]
        module._parameters[attr] = torch.nn.Parameter(torch.empty(0, dtype=p.dtype, device="meta"), requires_grad=False)

    def load_persistent(self):
        lm = self.model.language_model
        for entry in (*self.parameters_of(lm.embed_tokens, "language_model.embed_tokens"),
                      *self.parameters_of(lm.norm, "language_model.norm")):
            self.install(*entry)

    def attach(self):
        for index, layer in enumerate(self.model.language_model.layers):
            entries = list(self.parameters_of(layer, f"language_model.layers.{index}"))

            def pre(module, args, kwargs, entries=entries):
                for entry in entries:
                    self.install(*entry)
                return args, kwargs

            def post(module, args, output, entries=entries):
                for m, attr, _ in entries:
                    self.evict(m, attr)

            self.handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(post))

    def detach(self):
        for h in self.handles:
            h.remove()


def measure(checkpoint_root, g, device, gdn_kernel="reference", log=print):
    """(site -> amax, tokens, the GDN rule's implementation): one prefill of the corpus through the
    checkpoint's text model."""
    from accelerate import init_empty_weights
    kernel = gdn_reference() if gdn_kernel == "reference" else "installed"
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5Model

    config = AutoConfig.from_pretrained(checkpoint_root, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_root)
    with init_empty_weights(include_buffers=False):
        model = Qwen3_5Model(config)
    model.eval()
    site_modules = sites(g)
    modules = dict(model.named_modules())
    peaks = {name: 0.0 for name in site_modules}

    def capture(name):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            peaks[name] = max(peaks[name], hidden.detach().abs().amax().item())
        return hook

    for name, path in site_modules.items():
        if path not in modules:
            raise KeyError(f"the model has no module {path} for {name}")
        modules[path].register_forward_pre_hook(capture(name), with_kwargs=True)

    ids = tokenizer(corpus(), return_tensors=None)["input_ids"]
    with shards.ShardReader.checkpoint(checkpoint_root) as reader:
        layers = StreamingLayers(model, reader, device)
        layers.load_persistent()
        for module in model.modules():
            for b, buf in module.named_buffers(recurse=False):
                if buf.device.type == "cpu" and buf.device != torch.device(device):
                    module._buffers[b] = buf.to(device)
        layers.attach()
        t0 = time.time()
        with torch.inference_mode():
            model(torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        layers.detach()
    log(f"calibration prefill: {len(ids)} tokens in {time.time() - t0:.1f} s")
    return peaks, len(ids), kernel


def divisors(peaks):
    """site -> {amax, input_scale_divisor}; a site that saw no input is an error."""
    out = {}
    for name, amax in peaks.items():
        if not amax > 0.0:
            raise ValueError(f"{name} saw no nonzero input over the corpus")
        out[name] = {"amax": amax, "input_scale_divisor": encode.divisor_of(amax)}
    return out
