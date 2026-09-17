# ignis -- developer Makefile.
#
#   make                 the target list (same as `make help`)
#   make dev             build ignis-server, then run it (GPU, release, Playground)
#   make run             run the last build; refuses a missing or stale binary
#   make mock            the same loop on the CPU mock (no GPU, no kernel leaf)
#   make dev-ui          server in the background + Playground with hot reload
#                        (API_KEY=auto: the server generates a key and prints it;
#                        EXPOSE=cloudflare-quick: a public URL, key required)
#   make config          the resolved knobs
#
# Layout
#   Makefile             the platform-neutral target graph (this file)
#   mk/config.mk         the knobs and their defaults (CUDA, PROFILE, UI, ...)
#   mk/os/<os>.mk        the per-OS hooks: target triple, kernel build, GPU
#                        checks, background server control. Windows is
#                        implemented; Linux is scaffolded and every hook it
#                        cannot run yet fails with a "not implemented" message.
#   mk/windows/*.ps1     the PowerShell helpers behind the Windows hooks
#   local.mk             untracked per-machine overrides (see local.mk.example)
#
# Precedence: command line > environment > local.mk > mk/config.mk.
#
# Recipes run under a POSIX sh: /bin/sh on Linux, Git for Windows' sh.exe on
# Windows (GNU make takes it from PATH). Keep them to plain sh + the tools
# listed by `make doctor`.
#
# There is deliberately no `fmt` target: the repo is not rustfmt-clean and
# diffs stay semantic-only.

MAKEFLAGS += --no-builtin-rules --no-builtin-variables
.SUFFIXES:
.DEFAULT_GOAL := help
# cargo, ninja and vite parallelize internally; make itself stays serial so a
# `run` never races its own freshness check or the GPU guard.
.NOTPARALLEL:

-include local.mk
include mk/config.mk

ifeq ($(OS),Windows_NT)
  HOST_OS := windows
else
  HOST_OS := $(shell uname -s | tr '[:upper:]' '[:lower:]')
endif
ifeq ($(wildcard mk/os/$(HOST_OS).mk),)
  $(error unsupported host OS '$(HOST_OS)': add mk/os/$(HOST_OS).mk (start from mk/os/linux.mk))
endif
include mk/os/$(HOST_OS).mk

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------

ifeq ($(PROFILE),release)
  PROFILE_DIR := release
else ifeq ($(PROFILE),dev)
  PROFILE_DIR := debug
else
  PROFILE_DIR := $(PROFILE)
endif

ifeq ($(CUDA),1)
  BACKEND := cuda
else
  BACKEND := cpu
endif

# `--target` is always explicit: .cargo/config.toml pins the MSVC triple for
# Windows, and the flag is what lets the same graph target Linux later.
CARGO_TARGET := --target $(TARGET_TRIPLE)
SERVER_CARGO_FLAGS := -p ignis-server $(CARGO_TARGET) --profile $(PROFILE) $(if $(filter 1,$(CUDA)),--features cuda)
WORKSPACE_FEATURES := $(if $(filter 1,$(CUDA)),--features ignis-server/cuda)

BIN_DIR := $(CARGO_TARGET_DIR)/$(TARGET_TRIPLE)/$(PROFILE_DIR)
SERVER_BIN := $(BIN_DIR)/ignis-server$(EXE)
# Records the backend of the last `make build`, so `make run CUDA=1` cannot
# silently start a binary that was built as the CPU mock (or the reverse).
SERVER_STAMP := $(BIN_DIR)/ignis-server.backend
BUILT_BACKEND = $(strip $(if $(wildcard $(SERVER_STAMP)),$(file < $(SERVER_STAMP))))

SERVER_URL := http://$(BIND)
# The server's own default when METRICS_BIND is empty (DEFAULT_METRICS_BIND).
METRICS_URL := http://$(or $(METRICS_BIND),127.0.0.1:9464)
SERVER_PID := $(RUNTIME_DIR)/ignis-server.pid
SERVER_LOG := $(RUNTIME_DIR)/ignis-server.log

GPU_ENGINE_FLAGS = $(if $(ARTIFACT),--artifact $(ARTIFACT)) \
  $(if $(KV_FORMAT),--kv-format $(KV_FORMAT)) \
  $(if $(MAX_CONTEXT),--max-context $(MAX_CONTEXT)) \
  $(if $(PREFILL_CHUNK),--prefill-chunk $(PREFILL_CHUNK)) \
  $(if $(KV_POOL_BYTES),--kv-pool-bytes $(KV_POOL_BYTES))   $(if $(VRAM_HEADROOM),--vram-headroom-bytes $(VRAM_HEADROOM))   $(if $(VRAM_BUDGET),--vram-budget-bytes $(VRAM_BUDGET))   $(if $(filter 1,$(ALLOW_VRAM_OVERSUBSCRIPTION)),--allow-vram-oversubscription) \
  $(if $(KV_HOST_POOL_BYTES),--kv-host-pool-bytes $(KV_HOST_POOL_BYTES)) \
  $(if $(REQUEST_TIMEOUT),--request-timeout $(REQUEST_TIMEOUT)) \
  $(if $(SPEC),--spec $(SPEC) $(if $(DRAFT_TOKENS),--draft-tokens $(DRAFT_TOKENS)))
SERVER_FLAGS = --bind $(BIND) \
  $(if $(filter 1,$(CUDA)),$(GPU_ENGINE_FLAGS)) \
  $(if $(MODEL),--model $(MODEL)) \
  $(if $(filter 1,$(UI)),--ui) \
  $(if $(filter 1,$(METRICS)),--metrics $(if $(METRICS_BIND),--metrics-bind $(METRICS_BIND))) \
  $(if $(API_KEY),--api-key $(API_KEY)) \
  $(if $(EXPOSE),--expose $(EXPOSE)) \
  $(if $(SYSTEM_MESSAGE_POLICY),--system-message-policy $(SYSTEM_MESSAGE_POLICY)) \
  $(if $(DEVELOPER_MESSAGE_POLICY),--developer-message-policy $(DEVELOPER_MESSAGE_POLICY)) \
  $(ARGS)
SERVER_ENV = $(if $(LOG_LEVEL),IGNIS_LOG_LEVEL=$(LOG_LEVEL)) $(if $(LOG_FORMAT),IGNIS_LOG_FORMAT=$(LOG_FORMAT))
SMOKE_MODEL := $(or $(MODEL),qwen3.8-27b)
# A generated key (API_KEY=auto) is unknown to make: pass the printed one.
SMOKE_AUTH := $(if $(filter-out auto,$(API_KEY)),-H 'Authorization: Bearer $(API_KEY)')

GPU_GUARDED := $(and $(filter 1,$(CUDA)),$(filter 1,$(GPU_CHECK)))

# $(call rfiles,<dir>): every file below <dir>, recursively (directories are
# descended, never listed: `<entry>/.` only exists for a directory).
rfiles = $(foreach e,$(wildcard $(1:=/*)),$(if $(wildcard $e/.),$(call rfiles,$e),$e))

# What the server binary is made of. Test trees are left out: they never
# reach the binary. web/dist is embedded by crates/server/build.rs. The
# kernel sources mirror the whitelist in crates/artifact/build.rs and only
# count for a CUDA build.
SERVER_CRATES := server core runtime artifact logging
SERVER_INPUTS := Cargo.toml $(wildcard .cargo/config.toml) \
  $(foreach c,$(SERVER_CRATES),crates/$c/Cargo.toml $(wildcard crates/$c/build.rs) $(call rfiles,crates/$c/src)) \
  $(call rfiles,web/dist)
ifeq ($(CUDA),1)
  SERVER_INPUTS += $(wildcard kernel/CMakeLists.txt) $(KERNEL_BUILD_INPUTS) \
    $(foreach r,src include vendor/src vendor/include,$(call rfiles,kernel/$r))
endif

WEB_INDEX := web/dist/index.html
WEB_DEPS := web/node_modules/.package-lock.json
WEB_INPUTS := $(call rfiles,web/src) \
  $(wildcard web/index.html web/package.json web/vite.config.ts web/tsconfig.json web/mock.ts)

# $(call stale_error,<what>,<file>,<newer inputs>,<fix>)
define stale_error
if [ -e "$2" ]; then \
  echo "error: $1 is out of date ($2)"; \
  echo "  changed since it was built:"; \
  for f in $(wordlist 1,8,$3); do echo "    $$f"; done; \
  $(if $(word 9,$3),echo "    ... and $(words $(wordlist 9,999999,$3)) more";) \
else \
  echo "error: $1 is not built ($2)"; \
fi; \
echo "  fix: $4"; \
exit 1
endef

# ---------------------------------------------------------------------------
##@ Help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this list
	@awk 'BEGIN { FS = ":.*## " } \
	  /^##@/ { printf "\n%s\n", substr($$0, 5); next } \
	  /^[a-zA-Z0-9_.%-]+:.*## / { printf "  %-16s %s\n", $$1, $$2 }' Makefile
	@echo ""
	@echo "Knobs (make config for all):  CUDA=$(CUDA)  PROFILE=$(PROFILE)  UI=$(UI)  METRICS=$(METRICS)  BIND=$(BIND)  host=$(HOST_OS)"
	@echo "GPU engine: MAX_CONTEXT=$(MAX_CONTEXT)  SPEC=$(or $(SPEC),off)  DRAFT_TOKENS=$(DRAFT_TOKENS)  KV_FORMAT=$(KV_FORMAT)"
	@echo "  e.g.  make dev CUDA=0 PROFILE=dev      make dev-ui SPEC= MAX_CONTEXT=40960"

.PHONY: config
config: ## Print the resolved knobs and paths
	@echo "host OS         $(HOST_OS)"
	@echo "target triple   $(TARGET_TRIPLE)"
	@echo "CUDA            $(CUDA)  (backend=$(BACKEND))"
	@echo "PROFILE         $(PROFILE)  (dir=$(PROFILE_DIR))"
	@echo "UI              $(UI)"
	@echo "METRICS         $(METRICS)  $(if $(filter 1,$(METRICS)),(Prometheus: $(METRICS_URL)/metrics; Playground: /ui/metrics$(if $(or $(API_KEY),$(EXPOSE)), behind the API key)),(off))"
	@echo "ARTIFACT        $(ARTIFACT)"
	@echo "engine (CUDA=1) context=$(or $(MAX_CONTEXT),default) kv=$(or $(KV_FORMAT),default) chunk=$(or $(PREFILL_CHUNK),default) pool=$(or $(KV_POOL_BYTES),rest of the VRAM budget) host_pool=$(or $(KV_HOST_POOL_BYTES),default) timeout=$(or $(REQUEST_TIMEOUT),default) spec=$(or $(SPEC),off)$(if $(SPEC),/$(DRAFT_TOKENS))"
	@echo "VRAM (CUDA=1)   $(if $(VRAM_BUDGET),budget=$(VRAM_BUDGET)$(if $(filter 1,$(ALLOW_VRAM_OVERSUBSCRIPTION)), (oversubscription allowed)),headroom=$(or $(VRAM_HEADROOM),(server default: 1G)))"
	@echo "MODEL           $(or $(MODEL),(server default))"
	@echo "BIND            $(BIND)"
	@echo "LOG_LEVEL       $(or $(LOG_LEVEL),(server default))"
	@echo "LOG_FORMAT      $(or $(LOG_FORMAT),(server default))"
	@echo "API_KEY         $(if $(API_KEY),$(if $(filter auto,$(API_KEY)),auto (generated and printed at start),set),$(if $(EXPOSE),(none: auto, required by EXPOSE),(none: /v1 is open)))"
	@echo "EXPOSE          $(or $(EXPOSE),(none: reachable at BIND only))"
	@echo "MESSAGES        system=$(or $(SYSTEM_MESSAGE_POLICY),(server default: merge)) developer=$(or $(DEVELOPER_MESSAGE_POLICY),(server default: inplace))"
	@echo "ARGS            $(ARGS)"
	@echo "GPU_CHECK       $(GPU_CHECK)  (threshold $(GPU_THRESHOLD_MIB) MiB)"
	@echo "server binary   $(SERVER_BIN)"
	@echo "built backend   $(or $(BUILT_BACKEND),(unknown: not built by make))"
	@echo "server flags    $(strip $(SERVER_FLAGS))"
	@echo "runtime dir     $(RUNTIME_DIR)"

print-%: ## Print one make variable (make print-SERVER_BIN)
	@echo '$*=$($*)'

# ---------------------------------------------------------------------------
##@ Setup
# ---------------------------------------------------------------------------

.PHONY: doctor
doctor: ## Check the toolchain, the artifact and the web deps
	@missing=0; \
	check() { \
	  if command -v "$$1" >/dev/null 2>&1; then printf '  ok       %s\n' "$$1"; \
	  elif [ "$$2" = optional ]; then printf '  optional %s (%s)\n' "$$1" "$$3"; \
	  else printf '  MISSING  %s (%s)\n' "$$1" "$$3"; missing=1; fi; \
	}; \
	echo "tools:"; \
	check $(CARGO) required "https://rustup.rs"; \
	check rustup required "https://rustup.rs"; \
	check $(NPM) required "Node.js, for the Playground"; \
	check curl required "smoke / status checks"; \
	$(foreach t,$(OS_REQUIRED_TOOLS),check $(t) required "$(OS_TOOL_HINT)";) \
	$(foreach t,$(OS_OPTIONAL_TOOLS),check $(t) optional "$(OS_TOOL_HINT)";) \
	if $(CARGO) watch --version >/dev/null 2>&1; then echo "  ok       cargo-watch"; \
	else echo "  optional cargo-watch (make watch: cargo install cargo-watch)"; fi; \
	echo "rust target:"; \
	if rustup target list --installed 2>/dev/null | grep -qx "$(TARGET_TRIPLE)"; then echo "  ok       $(TARGET_TRIPLE)"; \
	else echo "  MISSING  $(TARGET_TRIPLE) (make setup)"; missing=1; fi; \
	echo "artifact:"; \
	if [ -f "$(ARTIFACT)" ]; then echo "  ok       $(ARTIFACT)"; \
	else echo "  missing  $(ARTIFACT) (only needed with CUDA=1)"; fi; \
	echo "web:"; \
	if [ -f "$(WEB_DEPS)" ]; then echo "  ok       web/node_modules"; \
	else echo "  missing  web/node_modules (make web-install)"; fi; \
	if [ -f "$(WEB_INDEX)" ]; then echo "  ok       web/dist"; \
	else echo "  missing  web/dist (make web-build; UI=1 builds do it for you)"; fi; \
	$(OS_DOCTOR) \
	exit $$missing

.PHONY: setup
setup: web-install ## Install the Rust target and the web dependencies
	rustup target add $(TARGET_TRIPLE)

# ---------------------------------------------------------------------------
##@ Build
# ---------------------------------------------------------------------------

.PHONY: build
build: $(if $(filter 1,$(UI)),web-build) ## Build ignis-server (CUDA=1 builds the kernel leaf too)
	$(CARGO) build $(SERVER_CARGO_FLAGS)
	@printf '%s\n' "$(BACKEND)" > "$(SERVER_STAMP)"
	@echo "built $(SERVER_BIN)  (backend=$(BACKEND), profile=$(PROFILE), ui=$(if $(wildcard $(WEB_INDEX)),embedded,absent))"

.PHONY: build-all
build-all: ## Build every workspace binary (server, bench, inspect, vendor)
	$(CARGO) build --workspace $(CARGO_TARGET) --profile $(PROFILE) $(WORKSPACE_FEATURES)

.PHONY: kernel
kernel: ## Build the C++/CUDA kernel leaf only
	$(KERNEL_BUILD)

.PHONY: release
release: ## Clean-slate release build: web + GPU server
	$(MAKE) --no-print-directory web-build
	$(MAKE) --no-print-directory build PROFILE=release CUDA=1 UI=1

# ---------------------------------------------------------------------------
##@ Run
# ---------------------------------------------------------------------------

.PHONY: run
run: _require-built $(if $(GPU_GUARDED),gpu-guard) ## Run the last build in the foreground (never builds)
	$(SERVER_ENV) $(SERVER_BIN) $(SERVER_FLAGS)

.PHONY: dev
dev: ## Build, then run in the foreground (stops at a build error)
	$(MAKE) --no-print-directory build
	$(MAKE) --no-print-directory run

.PHONY: mock
mock: ## `make dev` on the CPU mock: no GPU, no kernel leaf, no artifact
	$(MAKE) --no-print-directory dev CUDA=0

.PHONY: start
start: _require-built $(if $(GPU_GUARDED),gpu-guard) ## Daemon: start in the background, outlives the terminal until make stop
	@mkdir -p "$(RUNTIME_DIR)"
	@$(SERVER_ENV) $(SERVER_START)

.PHONY: stop
stop: ## Stop the background server (FORCE=1 also stops one make did not start)
	@$(SERVER_STOP)

.PHONY: restart
restart: ## Stop, then start the last build again (no build)
	$(MAKE) --no-print-directory stop
	$(MAKE) --no-print-directory start

.PHONY: redeploy
redeploy: ## Build, and only if it succeeds: stop + start
	$(MAKE) --no-print-directory build
	$(MAKE) --no-print-directory stop
	$(MAKE) --no-print-directory start

.PHONY: status
status: ## Is the server up? (process, pid file, /v1/models)
	-@$(SERVER_STATUS)
	@code=$$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$(SERVER_URL)/v1/models" 2>/dev/null); \
	case "$$code" in \
	  200) echo "http: $(SERVER_URL) is serving";; \
	  401) echo "http: $(SERVER_URL) is serving (API key required)";; \
	  *) echo "http: nothing answering on $(SERVER_URL)";; \
	esac

.PHONY: logs
logs: ## Follow the background server log
	@[ -f "$(SERVER_LOG)" ] || { echo "no log at $(SERVER_LOG) (make start writes it)"; exit 1; }
	tail -n 100 -f "$(SERVER_LOG)"

.PHONY: smoke
smoke: ## One /v1/models + one short chat completion against BIND
	curl -fsS $(SMOKE_AUTH) "$(SERVER_URL)/v1/models"
	@echo ""
	curl -fsS $(SMOKE_AUTH) "$(SERVER_URL)/v1/chat/completions" -H 'Content-Type: application/json' \
	  -d '{"model":"$(SMOKE_MODEL)","messages":[{"role":"user","content":"Say hi in five words."}],"max_tokens":32}'
	@echo ""

.PHONY: metrics
metrics: ## Scrape the metrics listener (a server started with METRICS=1; no key)
	curl -fsS "$(METRICS_URL)/metrics"

.PHONY: canary
canary: ## ignis-bench canary suite against the running server
	$(CARGO) run -p ignis-bench $(CARGO_TARGET) -- canary --endpoint $(SERVER_URL)

# Freshness guard for run/start: fails with the reason instead of building.
# The sub-make's own "*** Error" trailer is dropped; the reason is on stdout.
.PHONY: _require-built
_require-built:
	@$(MAKE) --no-print-directory STRICT=1 _fresh 2>/dev/null || { echo "(run/start never build: see the fix above)"; exit 1; }

.PHONY: _fresh
_fresh: $(SERVER_BIN) $(if $(filter 1,$(UI)),$(if $(wildcard $(WEB_INDEX)),$(WEB_INDEX))) $(if $(filter 1,$(CUDA)),_require-artifact)
	@built='$(BUILT_BACKEND)'; \
	if [ -z "$$built" ]; then \
	  echo "warning: $(SERVER_BIN) was not built by make; cannot confirm it is the $(BACKEND) backend"; \
	elif [ "$$built" != "$(BACKEND)" ]; then \
	  echo "error: $(SERVER_BIN) was built with backend=$$built, but CUDA=$(CUDA) wants $(BACKEND)"; \
	  echo "  fix: make build CUDA=$(CUDA)   (or make dev)"; \
	  exit 1; \
	fi

$(SERVER_BIN): $(SERVER_INPUTS)
	@$(call stale_error,ignis-server,$@,$?,make build   (or make dev to build and run))

.PHONY: _require-artifact
_require-artifact:
	@[ -f "$(ARTIFACT)" ] || { echo "error: artifact not found: $(ARTIFACT)"; echo "  fix: ARTIFACT=<path.ninfer>, or CUDA=0 for the CPU mock"; exit 1; }

# ---------------------------------------------------------------------------
##@ Web (Playground)
# ---------------------------------------------------------------------------

.PHONY: web-install
web-install: $(WEB_DEPS) ## npm ci, only when package-lock.json changed

.PHONY: web-build
web-build: $(WEB_INDEX) ## Build web/dist, only when web sources changed

# Vite runs as `node vite.js`, never through npm: on Windows make may start
# npm.cmd, and a batch file answers Ctrl+C with "Terminate batch job (Y/N)?"
# instead of exiting.
# $(call vite,<env assignments>,<vite args>)
vite = cd web && $1 exec node node_modules/vite/bin/vite.js $2

.PHONY: web-dev
web-dev: $(WEB_DEPS) ## Vite with hot reload, proxying /v1 to the server on BIND
	$(call vite,IGNIS_URL=$(SERVER_URL))

.PHONY: web-mock
web-mock: $(WEB_DEPS) ## Vite with hot reload against the in-process fake engine
	$(call vite,,--mode mock)

.PHONY: dev-ui
dev-ui: ## Build, then run-ui (Ctrl+C stops the server and Vite together)
	$(MAKE) --no-print-directory build
	$(MAKE) --no-print-directory run-ui

.PHONY: run-ui
run-ui: _require-built $(if $(GPU_GUARDED),gpu-guard) $(WEB_DEPS) ## Last build + Vite hot reload as one session; Ctrl+C stops both
	@mkdir -p "$(RUNTIME_DIR)"
	@$(SERVER_ENV) $(DEV_UI)

.PHONY: watch
watch: ## Rebuild + restart the server on every Rust change (cargo-watch; best with CUDA=0)
	@$(CARGO) watch --version >/dev/null 2>&1 || { echo "error: cargo-watch not installed"; echo "  fix: cargo install cargo-watch"; exit 1; }
	@$(if $(filter 1,$(CUDA)),echo "note: CUDA=1 reloads the whole artifact into VRAM on every change; CUDA=0 is the fast loop")
	$(if $(GPU_GUARDED),$(MAKE) --no-print-directory gpu-guard)
	$(SERVER_ENV) $(CARGO) watch --why -w crates -w Cargo.toml -w web/dist \
	  $(if $(filter 1,$(CUDA)),-w kernel/src -w kernel/include -w kernel/vendor/src -w kernel/vendor/include) \
	  -i 'crates/*/tests/**' \
	  -x "run $(SERVER_CARGO_FLAGS) -- $(strip $(SERVER_FLAGS))"

$(WEB_DEPS): web/package.json $(wildcard web/package-lock.json)
	$(NPM) --prefix web ci
	@touch "$@"

$(WEB_INDEX): $(WEB_INPUTS) $(if $(STRICT),,| $(WEB_DEPS))
ifdef STRICT
	@$(call stale_error,the Playground build,$@,$?,make web-build   (then make build to embed it))
else
	$(NPM) --prefix web run build
	@touch "$@"
endif

# ---------------------------------------------------------------------------
##@ Test & quality
# ---------------------------------------------------------------------------

.PHONY: test
test: ## cargo test, workspace-wide (CPU only, never touches the GPU)
	$(CARGO) test --workspace $(CARGO_TARGET)

.PHONY: test-web
test-web: $(WEB_DEPS) ## Playground unit tests (vitest)
	$(NPM) --prefix web run test

.PHONY: typecheck-web
typecheck-web: $(WEB_DEPS) ## Playground TypeScript check
	$(NPM) --prefix web run typecheck

.PHONY: check
check: ## cargo check, every target of every crate
	$(CARGO) check --workspace --all-targets $(CARGO_TARGET) $(WORKSPACE_FEATURES)

.PHONY: clippy
clippy: ## cargo clippy, every target of every crate
	$(CARGO) clippy --workspace --all-targets $(CARGO_TARGET) $(WORKSPACE_FEATURES)

.PHONY: test-all
test-all: test typecheck-web test-web ## Everything CPU-side: cargo test + web typecheck + vitest

.PHONY: ci
ci: check test-all ## What a CI job would run (no GPU)

# ---------------------------------------------------------------------------
##@ GPU (one run on the card at a time -- docs/agents/testing.md)
# ---------------------------------------------------------------------------

.PHONY: gpu-status
gpu-status: ## VRAM in use + every process that may hold the card
	@$(GPU_STATUS)

.PHONY: gpu-guard
gpu-guard: ## Refuse when the GPU is held (ninfer, another ignis, a GPU test)
	@$(GPU_GUARD)

.PHONY: gpu-profile
gpu-profile: ## The explicit GPU test profile (GPU_PROFILE_ARGS=-SkipKernelBuild ...)
	$(GPU_PROFILE)

.PHONY: kernel-test
kernel-test: gpu-guard ## Kernel leaf op tests via CTest (needs the GPU)
	$(KERNEL_TEST)

# ---------------------------------------------------------------------------
##@ Clean
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove this profile's server binary and stamp (cheap)
	rm -f "$(SERVER_BIN)" "$(SERVER_STAMP)"

.PHONY: clean-web
clean-web: ## Remove web/dist
	rm -rf web/dist

.PHONY: clean-kernel
clean-kernel: ## Remove kernel/build (the next CUDA build reconfigures)
	rm -rf kernel/build

.PHONY: clean-runtime
clean-runtime: ## Remove the background server's pid file and logs
	rm -f "$(SERVER_PID)" "$(SERVER_LOG)" "$(SERVER_LOG).err"

.PHONY: distclean
distclean: clean-web clean-kernel clean-runtime ## Everything: cargo clean + web deps + kernel build
	$(CARGO) clean
	rm -rf web/node_modules
