# --- shared
DATA_DIR    ?= data
SEED        ?= 42
# sampling seed forwarded to the API (empty = not sent)
LLM_SEED    ?=
MAX_COMPLETION_TOKENS ?= 256000

# --- download-data
# TEXT_ONLY=0 keeps image/audio/video questions
TEXT_ONLY   ?= 1

# --- run-verbosity-analysis
MODEL_ID    ?=
TOP_P       ?= 0.95
TEMPERATURE ?= 1.0
# openrouter: none | low | mid | high | max ; budget: on | off
THINKING    ?= none
# openrouter | budget
API_STYLE   ?= openrouter
# openrouter only: provider slug(s), comma-separated; fallbacks are always disabled
PROVIDER    ?= wafer/us
# budget only: endpoint path appended to LLM_BASE_URL (default: agent)
ENDPOINT    ?=
# budget only: budget_tokens sent when THINKING=on
THINKING_BUDGET ?=
RETRIES     ?= 5
# per-request timeout in seconds (not streamed!)
TIMEOUT     ?= 3600
WORKERS     ?= 1
OUTPUT_DIR  ?= llm_verbosity_stats
# subset: LIMIT=10 -> first 10 per dataset; add OFFSET=20 to skip, or SUBSET_SEED=1 for a random pick
LIMIT       ?= 2
OFFSET      ?= 0
SUBSET_SEED ?= 1
# e.g. ONLY="MATH-500 hle-diamond" or ONLY="easy hard"
ONLY        ?=easy
# VERBOSE=1 enables --verbose
VERBOSE     ?= 1
EXTRA_ARGS  ?=

.PHONY: download-data run-verbosity-analysis

download-data:
	uv run python -m llm_verbosity.filter_data.prepare \
		--output-dir $(DATA_DIR) --seed $(SEED) \
		$(if $(filter 0,$(TEXT_ONLY)),--no-text-only,--text-only) \
		$(EXTRA_ARGS)

run-verbosity-analysis:
	@test -n "$(MODEL_ID)" || (echo "MODEL_ID is required, e.g. make run-verbosity-analysis MODEL_ID=my-model"; exit 1)
	uv run python -m llm_verbosity.run_analysis \
		--model-id "$(MODEL_ID)" \
		--top-p $(TOP_P) \
		--temperature $(TEMPERATURE) \
		--thinking $(THINKING) \
		--api-style $(API_STYLE) \
		$(if $(filter openrouter,$(API_STYLE)),--provider $(PROVIDER)) \
		$(if $(ENDPOINT),--endpoint $(ENDPOINT)) \
		$(if $(THINKING_BUDGET),--thinking-budget $(THINKING_BUDGET)) \
		$(if $(LLM_SEED),--seed $(LLM_SEED)) \
		--max-completion-tokens $(MAX_COMPLETION_TOKENS) \
		--retries $(RETRIES) \
		--timeout $(TIMEOUT) \
		--workers $(WORKERS) \
		--data-dir $(DATA_DIR) \
		--output-dir $(OUTPUT_DIR) \
		$(if $(LIMIT),--limit $(LIMIT)) \
		$(if $(OFFSET),--offset $(OFFSET)) \
		$(if $(SUBSET_SEED),--subset-seed $(SUBSET_SEED)) \
		$(if $(ONLY),--datasets $(ONLY)) \
		$(if $(VERBOSE),--verbose) \
		$(EXTRA_ARGS)
