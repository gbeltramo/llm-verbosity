# --- shared
DATA_DIR    ?= data
SEED        ?= 42

# --- download-data
# TEXT_ONLY=0 keeps image/audio/video questions
TEXT_ONLY   ?= 1

# --- run-verbosity-analysis
MODEL_ID    ?=
TOP_P       ?= 0.95
TEMPERATURE ?= 1.0
# none | low | medium (mid) | high | max
THINKING    ?= none
# openai | budget
API_STYLE   ?= budget
WORKERS     ?= 2
# subset: LIMIT=10 -> first 10 per dataset; add OFFSET=20 to skip, or SUBSET_SEED=1 for a random pick
LIMIT       ?= 2
OFFSET      ?= 0
SUBSET_SEED ?= 1
# e.g. ONLY="MATH-500 hle-diamond" or ONLY="easy hard"
ONLY        ?=
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
		--workers $(WORKERS) \
		--data-dir $(DATA_DIR) \
		$(if $(LIMIT),--limit $(LIMIT)) \
		$(if $(OFFSET),--offset $(OFFSET)) \
		$(if $(SUBSET_SEED),--subset-seed $(SUBSET_SEED)) \
		$(if $(ONLY),--datasets $(ONLY)) \
		$(if $(VERBOSE),--verbose) \
		$(EXTRA_ARGS)
