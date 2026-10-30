# llm-verbosity

Measure how verbose an LLM API is: **elapsed time, thinking tokens, answer (post-thinking)
tokens, generated tokens (thinking + answer) and total tokens**, on easy / mid / hard Q&A.

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

- `--api-style openrouter` (default): POSTs to `https://openrouter.ai/api/v1/chat/completions`,
  pins the provider with `"provider": {"only": ["wafer/us"], "allow_fallbacks": false}` and asks
  for per-request usage accounting with `"usage": {"include": true}`. Thinking is controlled with
  `"reasoning": {"enabled": ..., "effort": ...}` (`none|low|mid|high|max`; `mid` = `medium`).
- `--api-style budget`: POSTs to `{LLM_BASE_URL}/agent`; `THINKING=on` adds
  `"thinking": {"type": "enabled", "budget_tokens": N}` (default 4096, override with
  `THINKING_BUDGET` / `--thinking-budget`), `THINKING=none` omits it.
- The response is read exactly as the endpoint sends it: a JSON body (OpenRouter) or an SSE stream
  (the agent API streams even with `"stream": false`). The parser keeps the **final
  `message_delta`** (`usage` + `naivert.timings`) and the final `agent_done` messages.
- Requests run on `n` async workers and are retried with exponential backoff on transport errors,
  408/409/425/429 and 5xx. Reported time = the successful attempt only
  (`wall_with_retries_s` is stored in the JSON too).
- `--verbose` logs, per question, the first/last 30 chars of thinking and completion with the
  dataset tag and the question index (index inside the parquet file, also with subsets).

Output: a stats table + warnings summary on stdout, and a JSON in

```
./llm_verbosity_stats/{model_id}_{provider|budget}_{top_p}_{temp}_thinking_{...}.json
```

## What is measured

| Metric              | Meaning                                                                       |
|---------------------|-------------------------------------------------------------------------------|
| `elapsed_s`         | client-measured end-to-end request time                                       |
| `tokens_per_s`      | all generated tokens per second: the engine's `predicted_per_second` when reported (agent API), else `completion_tokens / elapsed_s` |
| `e2e_tokens_per_s`  | `completion_tokens / elapsed_s`, always                                       |
| `prompt_tokens`     | prompt + cached prompt tokens                                                 |
| `thinking_tokens`   | reasoning tokens (`output_tokens_details.thinking_tokens` / `completion_tokens_details.reasoning_tokens`) |
| `answer_tokens`     | `completion_tokens - thinking_tokens`                                         |
| `completion_tokens` | all generated tokens: thinking + answer                                       |
| `total_tokens`      | prompt + completion                                                           |

Everything comes from the API response (`usage` block, engine timings). Missing fields are shown
as `n/a` and reported as `missing_tokens`; nothing is estimated from text length. Table cells are
mean (median). `tokens_per_s_source` in the JSON says whether `tokens_per_s` is the engine's
`predicted_per_second` (`server`) or computed from the request time (`client`).

## Warnings

Logged live and summarized at the end (and per question in the JSON): missing usage block,
missing token counts, truncated generations (`finish_reason=length`), empty answers (budget spent
on thinking), retries, and permanent failures.

## Dev

```bash
uv run ruff check . && uv run ruff format .
```
