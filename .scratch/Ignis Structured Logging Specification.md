# Ignis Serve Structured Logging Specification

## 1. Purpose

Ignis MUST provide a single, application-wide structured logging system.

The logging architecture MUST satisfy all of the following goals:

- structured logging at the source;
- compatibility with the OpenTelemetry Logs data model;
- preference for OpenTelemetry Semantic Conventions where applicable;
- machine-readable JSONL output for production and automated ingestion;
- human-readable terminal output for interactive use;
- Twelve-Factor-compatible log streaming for long-running services;
- clean Unix CLI behavior for one-shot commands;
- no ad-hoc subsystem-specific logging formats;
- minimal impact on inference latency and throughput.

This specification applies to the entire Ignis application, including Rust, C/C++ host code, CUDA integration, FFI boundaries, HTTP/API handling, model execution, scheduling, and background work.

HTTP access logs are only one producer of the same application-wide logging system.

---

# 2. Core principle: structured at source

Every Ignis log entry MUST originate as a structured event.

Application code MUST NOT construct a preformatted textual line as the canonical representation of an event.

For example, application code SHOULD conceptually emit:

```text
event_name = "ignis.model.loaded"
severity = INFO
body = "Model loaded"

attributes:
  model.name = "Qwen3.8-27B"
  gpu.device.id = 0
  duration_ms = 17400
  memory.used_bytes = 22548578304
```

rather than:

```text
"Loaded Qwen3.8-27B on GPU 0 in 17.4s using 21 GiB"
```

as the only representation of the information.

The formatter or exporter decides how the event is represented.

Conceptually:

```text
                    +--> JSONL formatter
                    |
Structured event ---+--> Pretty formatter
                    |
                    +--> OpenTelemetry representation
```

The application subsystem producing the event MUST NOT need to know which representation is active.

---

# 3. Canonical event model

The canonical Ignis log event MUST map cleanly to an OpenTelemetry LogRecord.

The event model MUST support, where applicable:

- Timestamp
- ObservedTimestamp
- SeverityText
- SeverityNumber
- Body
- EventName
- Resource
- InstrumentationScope
- Attributes
- TraceId
- SpanId
- TraceFlags

Not every field is mandatory for every event.

Trace-related fields MUST only be populated when corresponding trace context exists.

The logging subsystem MUST NOT fabricate trace IDs or span IDs merely to populate logging fields.

---

# 4. Application-wide scope

All application-owned operational logging MUST use the canonical structured logging system.

This includes, but is not limited to:

- startup and shutdown;
- configuration loading;
- configuration validation;
- model discovery;
- model loading;
- tokenizer loading;
- tensor and weight loading;
- GPU discovery;
- CUDA initialization;
- CUDA errors;
- CUDA graph capture and replay;
- memory allocation;
- memory pressure;
- scheduler state changes;
- worker lifecycle;
- request lifecycle;
- HTTP/API operations;
- prefill;
- decode;
- MTP/speculative decoding;
- KV-cache lifecycle;
- batching;
- FFI boundaries;
- background tasks;
- recovery paths;
- errors and warnings;
- graceful shutdown.

Individual subsystems MUST NOT create incompatible logging formats or independent logging systems.

---

# 5. Output formats

Ignis MUST support at least:

```text
json
pretty
auto
```

Exact CLI spelling MAY follow established Ignis CLI conventions, but equivalent functionality MUST exist.

The selected output format changes only the presentation.

It MUST NOT change the underlying event semantics.

---

# 6. JSON output

`json` mode MUST produce machine-readable JSON Lines.

Requirements:

- UTF-8;
- one complete JSON object per physical line;
- newline terminates each record;
- no pretty-printed multiline JSON;
- no ANSI escape sequences;
- no arbitrary text before or after the JSON object;
- embedded newline characters MUST be JSON-escaped;
- stack traces MUST remain part of the same JSON record;
- every physical log line MUST be independently parseable as JSON.

Example:

```json
{"timestamp":"2026-09-07T13:42:17.123Z","severity_text":"INFO","severity_number":9,"event_name":"ignis.model.loaded","body":"Model loaded","attributes":{"model.name":"Qwen3.8-27B","gpu.device.id":0,"duration_ms":17400,"memory.used_bytes":22548578304}}
```

JSON mode MUST be appropriate for:

- Kubernetes;
- OpenShift;
- container runtimes;
- CI/CD;
- system services;
- log collectors;
- automated processing;
- redirected logs.

---

# 7. Pretty output

`pretty` mode MUST provide a human-readable representation of the same structured event.

Example:

```text
15:42:17.123  INFO   Model loaded
                     model=Qwen3.8-27B gpu=0 memory=21.0GiB duration=17.4s
```

or:

```text
15:42:17 INFO  Model loaded  model=Qwen3.8-27B gpu=0 memory=21.0GiB duration=17.4s
```

The exact visual representation MAY evolve.

Pretty mode MAY use:

- colors;
- indentation;
- aligned fields;
- abbreviated labels;
- human-readable sizes;
- human-readable durations;
- multiline rendering;
- shortened trace IDs.

ANSI colors SHOULD only be used when output is attached to an interactive terminal or when explicitly enabled.

Pretty output MUST originate from the same canonical structured event used by JSON output.

Application code MUST NOT emit separate "pretty" versions of events.

---

# 8. Auto mode

`auto` mode SHOULD select an appropriate formatter based on the runtime environment.

Recommended behavior:

```text
interactive TTY     -> pretty
non-interactive     -> json
```

For example:

```bash
ignis-serve
```

when launched interactively may select pretty output.

When launched in a container or with redirected output:

```bash
ignis-serve > ignis.log
```

auto mode may select JSONL.

Explicit user configuration MUST override auto detection.

Production deployments SHOULD explicitly configure their expected format rather than relying exclusively on TTY detection.

For example:

```text
IGNIS_LOG_FORMAT=json
```

---

# 9. Twelve-Factor logging behavior

Ignis MUST treat application logs as an event stream.

In long-running service mode:

```bash
ignis-serve
```

Ignis MUST:

- emit its application log stream to stdout;
- NOT manage log files;
- NOT implement application-owned log rotation;
- NOT implement application-owned retention;
- NOT choose persistent log storage;
- NOT require direct knowledge of the final logging backend;
- leave log capture, transport, aggregation, storage, indexing, and retention to the execution environment.

Conceptually:

```text
Ignis
  |
  | stdout
  v
container runtime / execution environment
  |
  v
logging infrastructure
  |
  +--> OpenTelemetry Collector
  +--> Loki
  +--> Elasticsearch
  +--> other backend
```

This applies independently of whether the selected formatter is:

```text
pretty
```

or:

```text
json
```

The log destination is determined by process role, not by formatting style.

---

# 10. stdout and stderr behavior

## 10.1 Long-running service

For:

```bash
ignis-serve
```

both JSON and pretty application logs SHOULD use stdout.

For example:

```text
stdout:
15:48:01 INFO  Model loaded
15:48:02 WARN  KV cache pressure
15:48:03 ERROR CUDA allocation failed
```

Severity MUST NOT by itself determine the output stream.

Ignis SHOULD NOT implement:

```text
INFO  -> stdout
WARN  -> stderr
ERROR -> stderr
```

for the long-running application event stream.

The goal is to keep the service log stream coherent and ordered.

---

## 10.2 One-shot CLI commands

For one-shot commands whose stdout contains a command result, stdout MUST remain usable as a data stream.

Examples:

```bash
ignis-serve inspect model
ignis-serve inspect model --output json
ignis-serve config validate
ignis-serve config dump
ignis-serve benchmark ...
```

For these commands, the recommended behavior is:

```text
stdout -> command result
stderr -> diagnostic logging
```

For example:

```bash
ignis-serve inspect model --output json | jq .
```

MUST NOT be broken by unrelated INFO or DEBUG records being written into stdout.

Diagnostics may therefore be written to stderr in one-shot command mode.

This exception does NOT change the underlying structured logging model.

---

# 11. Distinguish logs from command output

Structured logs and intentional CLI result data are different concepts.

For example:

```bash
ignis-serve inspect model --output json
```

may intentionally produce:

```json
{
  "model": "Qwen3.8-27B",
  "max_context_length": 262144
}
```

This is command output, not an application log record.

Application code MUST preserve this distinction.

Logging configuration MUST NOT unexpectedly corrupt machine-readable CLI output.

---

# 12. Configuration

Logging behavior MUST be configurable.

At minimum, Ignis MUST support configuration equivalent to:

```text
log format
log level
```

Recommended environment variables:

```text
IGNIS_LOG_FORMAT
IGNIS_LOG_LEVEL
```

Recommended values:

```text
IGNIS_LOG_FORMAT=auto
IGNIS_LOG_FORMAT=pretty
IGNIS_LOG_FORMAT=json
```

and:

```text
IGNIS_LOG_LEVEL=trace
IGNIS_LOG_LEVEL=debug
IGNIS_LOG_LEVEL=info
IGNIS_LOG_LEVEL=warn
IGNIS_LOG_LEVEL=error
```

Equivalent command-line flags SHOULD exist.

For example:

```bash
ignis-serve --log-format pretty
ignis-serve --log-format json
ignis-serve --log-level debug
```

Environment variables SHOULD remain available because deployment-dependent configuration must not depend exclusively on command-line arguments.

Command-line options MAY override environment configuration according to the project's established configuration precedence rules.

---

# 13. Severity

Ignis MUST support at least:

```text
TRACE
DEBUG
INFO
WARN
ERROR
```

Severity SHOULD map consistently to the OpenTelemetry severity model.

The logging subsystem MUST use stable severity semantics.

Example intent:

```text
TRACE
very high-frequency diagnostic information

DEBUG
developer/debugging information

INFO
normal meaningful lifecycle and operational events

WARN
unexpected condition from which Ignis can continue

ERROR
operation failed or important functionality could not complete
```

Application code SHOULD NOT misuse ERROR for expected control flow.

---

# 14. Event names

Operationally meaningful events SHOULD have stable machine-readable names.

Recommended pattern:

```text
ignis.<subsystem>.<event>
```

Examples:

```text
ignis.process.started
ignis.process.stopping
ignis.process.stopped

ignis.config.loaded
ignis.config.invalid

ignis.model.load.started
ignis.model.loaded
ignis.model.load.failed

ignis.gpu.discovered
ignis.gpu.initialized
ignis.gpu.initialization.failed

ignis.memory.allocation.failed
ignis.memory.pressure

ignis.cuda.graph.capture.started
ignis.cuda.graph.capture.completed
ignis.cuda.graph.capture.failed
ignis.cuda.graph.replay.failed

ignis.request.started
ignis.request.completed
ignis.request.failed

ignis.prefill.started
ignis.prefill.completed
ignis.prefill.failed

ignis.decode.started
ignis.decode.completed
ignis.decode.failed

ignis.worker.started
ignis.worker.stopped
ignis.worker.failed
```

Event names MUST identify an event class.

Bad:

```text
request-123-completed
qwen-loaded-on-gpu-zero
```

Good:

```text
ignis.request.completed
ignis.model.loaded
```

Occurrence-specific information belongs in attributes.

---

# 15. Body and attributes

The event body SHOULD contain a short human-readable description.

Example:

```text
body = "Model loaded"
```

Machine-readable information MUST be represented in structured attributes.

Example:

```text
model.name = "Qwen3.8-27B"
gpu.device.id = 0
duration_ms = 17400
memory.used_bytes = 22548578304
```

Consumers MUST NOT be required to parse the human-readable body to obtain important operational values.

Bad:

```text
body = "Model Qwen3.8-27B loaded on GPU 0 using 21GB in 17.4 seconds"
```

as the only representation.

Good:

```text
body = "Model loaded"

attributes:
  model.name = "Qwen3.8-27B"
  gpu.device.id = 0
  memory.used_bytes = 22548578304
  duration_ms = 17400
```

The pretty renderer MAY reconstruct a convenient human representation from those fields.

---

# 16. Resource attributes

Application identity and deployment identity SHOULD use OpenTelemetry Resource concepts.

Ignis SHOULD provide, when available:

```text
service.name
service.version
service.instance.id
```

with:

```text
service.name = "ignis"
```

Additional appropriate standard resource attributes MAY include:

```text
service.namespace
deployment.environment.name
host.*
process.*
container.*
k8s.*
```

depending on available runtime context.

Resource-level information SHOULD NOT be unnecessarily duplicated as event attributes.

---

# 17. OpenTelemetry Semantic Conventions

Ignis MUST prefer existing OpenTelemetry Semantic Convention attribute names whenever an appropriate standard attribute exists.

Ignis MUST NOT invent application-specific aliases for concepts already standardized by OpenTelemetry.

For example, prefer:

```text
service.name
```

over:

```text
app_name
ignis_service
application
service
```

Custom Ignis-specific attributes SHOULD use a stable namespace:

```text
ignis.*
```

Examples:

```text
ignis.scheduler.queue_depth
ignis.decode.sequence_id
ignis.cuda.graph.state
ignis.kv_cache.block_count
```

Ignis-specific attributes MUST NOT use reserved OpenTelemetry namespaces such as:

```text
otel.*
```

---

# 18. Attribute stability

Structured attribute names and their types form part of the observability contract.

Once an attribute becomes externally useful, its semantics SHOULD remain stable.

For example, if:

```text
ignis.scheduler.queue_depth
```

is an integer, it SHOULD NOT later become a formatted string such as:

```text
"17 requests"
```

Attributes SHOULD preserve native data types where possible.

Prefer:

```json
{"duration_ms":17400}
```

over:

```json
{"duration":"17.4 seconds"}
```

Human formatting belongs in the pretty renderer.

---

# 19. Trace correlation

When an active tracing context exists, the corresponding log event MUST be capable of carrying:

```text
TraceId
SpanId
TraceFlags
```

These identifiers MUST originate from the active trace context.

The logging subsystem MUST NOT independently generate replacement trace or span identifiers.

Example:

```json
{
  "event_name": "ignis.request.completed",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "attributes": {
    "request.id": "req-127",
    "duration_ms": 1761
  }
}
```

Events outside an active trace remain valid.

For example:

```text
ignis.process.started
ignis.gpu.initialized
ignis.model.loaded
```

MUST NOT require fabricated trace IDs.

Pretty output MAY omit trace IDs by default for readability.

Debug or verbose modes MAY display them.

---

# 20. OpenTelemetry compatibility

Ignis MUST be OpenTelemetry-compatible at the logging data-model level.

A canonical Ignis structured event SHOULD be convertible to an OpenTelemetry LogRecord without parsing its rendered text.

No implementation SHOULD need to parse:

```text
pretty console logs
```

or:

```text
JSON console logs
```

to reconstruct the original structured event.

The structured event is the source of truth.

---

# 21. OpenTelemetry transport

Direct OTLP log export from Ignis MUST NOT be required for logging compliance.

The preferred Twelve-Factor deployment architecture is:

```text
Ignis
   |
   stdout
   |
   v
execution environment
   |
   v
OpenTelemetry Collector or logging agent
   |
   v
OTLP/backend
```

Ignis MAY support direct OpenTelemetry log export in the future if there is a concrete requirement, but application logging MUST NOT depend on direct backend connectivity.

The normal service MUST remain fully functional when only stdout logging is configured.

Log routing, retention, backend selection, and transport belong primarily to the deployment environment.

---

# 22. Error logging

Errors MUST remain structured events.

Where appropriate, error information SHOULD use OpenTelemetry-compatible exception fields such as:

```text
exception.type
exception.message
exception.stacktrace
```

Example canonical event:

```text
event_name = "ignis.model.load.failed"
severity = ERROR
body = "Model loading failed"

attributes:
  model.name = "Qwen3.8-27B"
  gpu.device.id = 0
  exception.type = "CudaOutOfMemory"
  exception.message = "device allocation failed"
  memory.requested_bytes = 4294967296
```

Pretty output may show:

```text
15:45:33 ERROR Model loading failed
               model=Qwen3.8-27B gpu=0
               CudaOutOfMemory: device allocation failed
```

JSON output MUST preserve the event as one JSONL record.

---

# 23. Stack traces

Stack traces MAY be multiline in human-readable pretty output.

For example:

```text
ERROR Model loading failed
      CudaOutOfMemory: device allocation failed
        at ...
        at ...
```

In JSON mode, the stack trace MUST remain within one JSON object.

Physical JSONL framing MUST NOT be broken.

---

# 24. Request logging

HTTP/API logging MUST use the same canonical logging model.

Request logs SHOULD capture useful structured metadata such as:

```text
request.id
HTTP method
route
status code
duration
input token count
output token count
model
trace context
```

where appropriate.

HTTP logging MUST NOT introduce a separate unrelated access-log format unless required for compatibility with an external component.

Where OpenTelemetry Semantic Conventions exist, standard attributes SHOULD be preferred.

---

# 25. Inference-specific logging

Ignis SHOULD expose structured observability for important inference lifecycle events.

Useful categories MAY include:

```text
model load
model unload
request scheduling
batch formation
prefill
decode
MTP
KV cache
CUDA graph capture
CUDA graph replay
GPU memory pressure
worker lifecycle
queue pressure
```

However, observability MUST NOT turn into uncontrolled hot-path logging.

---

# 26. Hot-path logging constraints

Logging MUST NOT materially degrade inference performance.

Normal INFO-level operation MUST NOT produce one record:

- per generated token;
- per attention layer;
- per transformer layer;
- per CUDA kernel launch;
- per memory copy;
- per low-level allocation;
- per internal scheduler iteration.

High-frequency diagnostic information SHOULD use:

```text
TRACE
DEBUG
```

and SHOULD be disabled during normal production operation.

Where appropriate, high-frequency information SHOULD be:

- aggregated;
- sampled;
- exposed as metrics;
- exposed through tracing;
- emitted only on state transitions.

For example, prefer:

```text
KV cache pressure crossed 90%
```

over logging every individual KV-cache allocation.

---

# 27. Logging and synchronous I/O

Latency-sensitive inference paths SHOULD avoid uncontrolled blocking console I/O.

The implementation MAY use an internal logging mechanism that decouples event creation from physical output where necessary for performance.

However:

- logs MUST remain timely;
- queues SHOULD be bounded;
- uncontrolled memory growth MUST NOT be possible;
- important ERROR/WARN events SHOULD NOT be silently discarded;
- graceful shutdown SHOULD flush pending important events where practical.

Implementation details MUST NOT change the external structured logging contract.

---

# 28. Process shutdown

Ignis MUST handle graceful shutdown consistently with its logging lifecycle.

Important lifecycle events SHOULD include:

```text
ignis.process.stopping
ignis.process.stopped
```

On graceful termination, pending important logs SHOULD be flushed before process exit where practical.

The logging system MUST NOT introduce indefinite shutdown blocking.

---

# 29. Sensitive information

Ignis MUST NOT log secrets or credentials.

This includes, but is not limited to:

```text
API keys
Authorization headers
Bearer tokens
cookies
passwords
private credentials
secret environment variables
private keys
```

Prompts, completions, model input, tool arguments, and other user-provided content MUST NOT be logged by default.

Operational metadata MAY be logged when useful, including:

```text
request ID
model
token counts
latency
queue depth
GPU ID
memory usage
cache statistics
batch size
execution mode
```

---

# 30. Cardinality

Logging attributes SHOULD avoid unnecessary high-cardinality data when it provides little operational value.

High-cardinality identifiers MAY be included where needed for correlation, such as:

```text
request.id
trace_id
span_id
```

but SHOULD NOT be indiscriminately attached to every unrelated event.

Very large payloads MUST NOT be embedded in logs merely for debugging convenience.

---

# 31. No ad-hoc production logging

Production application code MUST NOT bypass the canonical logging subsystem with arbitrary diagnostic output.

Examples that MUST NOT be used as normal production logging:

```text
println!
eprintln!
printf
fprintf
std::cout
std::cerr
```

Exceptions are allowed when these primitives are intentionally used by:

- the centralized logging formatter;
- explicit CLI result output;
- bootstrapping before the logging system can technically initialize;
- explicitly gated development diagnostics.

Any bootstrap fallback SHOULD be minimal and SHOULD transition to the canonical logging system as early as possible.

---

# 32. Rust logging

Rust components MUST use the project's canonical structured logging API.

The implementation MAY use an established Rust logging/tracing framework if it satisfies this specification.

Subsystems MUST NOT create incompatible independent subscribers or formatters.

Structured fields MUST remain structured rather than being interpolated only into textual strings.

Prefer conceptually:

```text
info!(
    model = model_name,
    gpu_id = gpu_id,
    duration_ms = duration_ms,
    "Model loaded"
)
```

over:

```text
info!(
    "Model {} loaded on GPU {} in {} ms",
    model_name,
    gpu_id,
    duration_ms
)
```

when the former preserves fields structurally.

---

# 33. C/C++ and FFI logging

Ignis-owned C/C++ host components SHOULD integrate with the canonical Ignis logging system where feasible.

They MUST NOT silently introduce an unrelated text logging format.

FFI-related events SHOULD preserve structured information when crossing the boundary.

If a logging bridge is introduced, it SHOULD preserve:

```text
severity
event name
body
structured attributes
source subsystem
```

where practical.

---

# 34. CUDA device diagnostics

CUDA device-side `printf` MUST NOT be part of normal production logging.

Device-side diagnostics MAY exist for:

- development;
- GPU debugging;
- tests;
- explicitly enabled diagnostic builds.

Such diagnostics MUST be gated and MUST NOT become part of normal production output.

---

# 35. Third-party libraries

Third-party logs SHOULD be integrated into the common logging pipeline where practical.

If a third-party library emits unstructured messages that cannot be changed, Ignis MAY wrap them as structured events.

For example:

```text
event_name = "ignis.dependency.log"
body = "<original third-party message>"

attributes:
  dependency.name = "..."
```

Ignis MUST NOT pretend that arbitrary unstructured third-party text has structured semantics that are not actually available.

---

# 36. Source metadata

The logging implementation MAY attach source metadata such as:

```text
module
target
file
line
thread
task
```

where useful.

Verbose source metadata SHOULD normally be limited to DEBUG/TRACE or configuration that explicitly enables it if it creates excessive noise.

---

# 37. Human-readable formatting

Pretty mode SHOULD optimize for operator readability.

The formatter MAY simplify:

```text
22548578304 bytes
```

into:

```text
21.0 GiB
```

and:

```text
17400 ms
```

into:

```text
17.4s
```

without modifying the canonical typed values.

The structured event MUST retain machine-appropriate units.

---

# 38. Time representation

Canonical timestamps MUST be precise enough for request and inference diagnostics.

JSON output SHOULD use an unambiguous UTC timestamp representation, preferably RFC 3339 with sub-second precision.

Example:

```text
2026-09-07T13:42:17.123456Z
```

Pretty output MAY render local or shorter time representations when suitable for interactive use.

Changing visual timezone representation MUST NOT change the canonical event timestamp.

---

# 39. Testing requirements

Automated tests MUST cover the logging contract.

At minimum, tests MUST verify:

1. canonical events preserve structured typed fields;
2. JSON mode produces valid JSON;
3. JSON mode produces one physical line per event;
4. embedded newline characters do not break JSONL framing;
5. stack traces do not break JSONL framing;
6. pretty mode is generated from the same event model;
7. changing formatter does not change event semantics;
8. severity mapping is correct;
9. event names are preserved;
10. trace IDs and span IDs are included when active trace context exists;
11. trace IDs are not fabricated when no context exists;
12. sensitive values are not emitted by covered paths;
13. one-shot CLI machine-readable stdout is not polluted by diagnostic logs;
14. `ignis-serve` uses stdout for the application event stream;
15. JSON and pretty service logging use the same destination semantics;
16. explicit format configuration overrides auto detection;
17. structured attributes retain native types;
18. multiline pretty errors remain valid single-record JSON when rendered through JSON mode.

---

# 40. Static and review safeguards

The project SHOULD make accidental regression toward ad-hoc logging difficult.

Code review and/or automated checks SHOULD detect newly introduced application-owned uses of:

```text
println!
eprintln!
printf
fprintf
std::cout
std::cerr
```

when used as production diagnostics.

Legitimate CLI output and logging infrastructure implementation MUST not be incorrectly prohibited.

---

# 41. Acceptance criteria

The logging implementation is complete when all of the following are true:

- Ignis has one canonical structured event model;
- all major application-owned subsystems use it;
- HTTP logging uses the same system;
- GPU/CUDA/FFI-related host logging uses the same logging contract;
- structured values are not encoded exclusively inside human-readable strings;
- event fields map cleanly to OpenTelemetry Logs concepts;
- standard OpenTelemetry Semantic Conventions are preferred when available;
- Ignis-specific fields use stable names;
- JSONL output is available;
- pretty human-readable output is available;
- auto selection is available;
- an explicit format override is available;
- long-running `ignis-serve` emits its event stream to stdout;
- WARN and ERROR are not automatically split to stderr in service mode merely because of severity;
- one-shot CLI commands can reserve stdout for command results;
- diagnostic logging for one-shot commands can use stderr;
- machine-readable CLI output remains safely pipeable;
- Ignis does not manage persistent log files, rotation, retention, or backend storage;
- normal logging does not require direct connectivity to an observability backend;
- OpenTelemetry correlation can be preserved when trace context exists;
- sensitive information is not logged by default;
- JSON records remain one physical line even for exceptions;
- hot-path logging does not materially interfere with inference performance;
- automated tests enforce the important parts of this contract.

---

# 42. Architectural summary

The intended architecture is:

```text
                         Ignis application
                                |
                                v
                    Canonical structured event
                                |
                  +-------------+-------------+
                  |                           |
                  v                           v
            Pretty formatter            JSON formatter
                  |                           |
                  +-------------+-------------+
                                |
                         selected stream
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
             long-running serve       one-shot command
                    |                       |
                 stdout             stderr for logs
                                            |
                                      stdout reserved
                                      for command result
```

For deployed service operation:

```text
Ignis
  |
  | stdout JSONL
  v
OpenShift / container runtime
  |
  v
logging agent / OpenTelemetry Collector
  |
  v
observability backend
```

The most important architectural rule is:

> Ignis logs are structured events first and formatted output second.

JSON, pretty terminal rendering, stdout, stderr, and future observability integrations are presentation and transport concerns layered on top of the same canonical event model.