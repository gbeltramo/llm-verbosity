# llm-verbosity

Measure how verbose an LLM API is: **elapsed time, thinking tokens, answer (post-thinking)
tokens, completion tokens and total tokens**, on easy / mid / hard Q&A.

| Tag            | Difficulty | Source                                  |
|----------------|------------|-----------------------------------------|
| `MATH-500`     | easy       | `HuggingFaceH4/MATH-500` (test split)   |
| `GPQA-Diamond` | mid        | `Idavidrein/GPQA` (`gpqa_diamond`)      |
| `HLE-Diamond`  | hard       | `cais/hle` (configurable, see below)    |

## Setup

```bash
uv sync
set -x HF_TOKEN ...        # GPQA and HLE are gated: accept their terms on the Hub first
set -x LLM_BASE_URL https://api.example.com/v1   # or the full .../chat/completions URL
set -x LLM_API_KEY ...
```

## 1. Prepare data (`filter_data`)

```bash
make download-data                # text-only (default)
make download-data TEXT_ONLY=0    # keep image/audio/video questions
```

Writes 100 seeded questions per dataset to `data/*.parquet` (zstd).

- With `--text-only` (default) every question with a non-empty image/audio/video column is
  dropped **before** sampling, so you still get 100 text-only questions. Rationale-side media
  (e.g. HLE `rationale_image`) is ignored. Disable with `--no-text-only`.
- GPQA options are shuffled deterministically and rendered as A-D.
- HLE: no official "diamond" subset is known to me, so this is a seeded random sample of
  `cais/hle`; use `--hle-repo` / `--hle-split` (via `EXTRA_ARGS`) to point at a specific one.

## 2. Run the analysis

```bash
make run-verbosity-analysis MODEL_ID=my-model TOP_P=0.95 TEMPERATURE=1.0 THINKING=high WORKERS=16
make run-verbosity-analysis MODEL_ID=my-model THINKING=none API_STYLE=budget VERBOSE=1

# subsets
make run-verbosity-analysis MODEL_ID=my-model LIMIT=10                    # first 10 per dataset
make run-verbosity-analysis MODEL_ID=my-model LIMIT=10 OFFSET=20          # questions 20..29
make run-verbosity-analysis MODEL_ID=my-model LIMIT=10 SUBSET_SEED=1      # reproducible random 10
make run-verbosity-analysis MODEL_ID=my-model ONLY="easy hard" LIMIT=5    # only some datasets
```

- `--api-style openai`: sends `reasoning_effort` (`none|low|medium|high|max`; `mid` = `medium`).
- `--api-style budget`: adds `"thinking": {"type": "enabled", "budget_tokens": N}` to the payload
  (low/medium/high/max = 2048/8192/16384/24576, override with `--thinking-budget`); `none`
  removes the field.
- Requests are streamed (SSE), run on `n` async workers and retried with exponential backoff on
  transport errors, 408/409/425/429 and 5xx. Reported time = the successful attempt only
  (`wall_with_retries_s` is stored in the JSON too).
- `--verbose` logs, per question, the first/last 30 chars of thinking and completion with the
  dataset tag and the question index (index inside the parquet file, also with subsets).

Output: a stats table + token provenance table + warnings summary on stdout, and a JSON in

```
./llm_verbosity_stats/{model_id}_{top_p}_{temp}_thinking_{none|low|medium|high|max}.json
```

## Token counts and provenance

| Metric              | Meaning                                              |
|---------------------|------------------------------------------------------|
| `thinking_tokens`   | reasoning tokens                                     |
| `answer_tokens`     | tokens produced **after** thinking (visible answer)  |
| `completion_tokens` | thinking + answer (everything the model generated)   |
| `total_tokens`      | prompt + completion                                  |

Every value is stored as `tokens.<metric>` with `token_sources.<metric>` in the JSON:

| Source      | Meaning                                                                       |
|-------------|-------------------------------------------------------------------------------|
| `api`       | reported verbatim by the API                                                  |
| `derived`   | exact arithmetic on API numbers (e.g. `answer = completion - thinking`)       |
| `estimated` | `chars/4` heuristic: **not exact**                                            |
| `assumed`   | based on a hypothesis (e.g. no thinking requested -> 0 thinking tokens)       |
| `missing`   | not available                                                                 |

The cleanest answer-only number is `answer_tokens` with source `api`/`derived`, which needs the
API to report thinking tokens (`completion_tokens_details.reasoning_tokens`, `reasoning_tokens`
or `thinking_tokens`). If it only reports completion tokens, `answer_tokens` is estimated from
the visible text and `thinking_tokens = completion - answer` (also flagged as estimated),
because streamed thinking may be summarized and would undercount.

In the table, `~` marks cells containing estimated/assumed values and `API-exact` is the number
of questions whose four key metrics are all `api`/`derived`.

## Warnings

Logged live and summarized at the end (and per question in the JSON): estimated or assumed
token counts, missing usage block, truncated generations (`finish_reason=length`), empty
answers (budget spent on thinking), suspected hidden thinking, `answer_tokens` inconsistent with
the visible text, retries, permanent failures, and thinking settings that appear to be
ignored by the endpoint.

## Dev

```bash
uv run ruff check . && uv run ruff format .
```
