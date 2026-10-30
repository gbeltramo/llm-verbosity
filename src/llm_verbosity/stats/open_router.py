import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import httpx
from omegaconf import DictConfig, OmegaConf

from .configs import BenchmarkResult, OpenRouterConfig, PendingGeneration
from .datasets import build_prompts

CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
GENERATIONS_URL = "https://openrouter.ai/api/v1/generation"


def create_client(api_key: str) -> httpx.Client:
    return httpx.Client(
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(300.0, connect=20.0),
        http2=True,
    )


def send_request(
    client: httpx.Client,
    prompt: str,
    config: OpenRouterConfig,
) -> tuple[dict, float]:
    """Send a non-streaming chat completion request."""
    payload = {
        "model": config.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_completion_tokens": 256000,
        "reasoning": {
            "enabled": config.thinking_effort != "none",
            "effort": config.thinking_effort,
            "exclude": False,
        },
        "usage": {"include": True},
        "provider": {
            "only": [config.inference_provider],
            "allow_fallbacks": False,
        },
        "stream": False,
        "seed": config.random_seed,
    }

    start = time.perf_counter()
    response = client.post(CHAT_COMPLETIONS_URL, json=payload)
    response.raise_for_status()
    elapsed = time.perf_counter() - start

    return response.json(), elapsed


def get_generation_stats(
    client: httpx.Client,
    generation_id: str,
    max_attempts: int = 32,
    initial_delay: float = 2.0,
) -> dict | None:
    """Fetch server-side stats, retrying while the record isn't ready (404)."""
    delay = initial_delay
    for _ in range(1, max_attempts + 1):
        response = client.get(GENERATIONS_URL, params={"id": generation_id})

        if response.status_code == 200:
            return response.json()["data"]

        if response.status_code != 404:
            response.raise_for_status()

        print(f"DEBUG: {delay=}")
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)

    return None


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def extract_token_counts(stats: dict | None, usage: dict) -> dict[str, int | None]:
    """Return native and OpenRouter-normalized counts separately (never mixed)."""
    stats = stats or {}

    return {
        # provider-native (only available from generation stats)
        "native_prompt": stats.get("native_tokens_prompt"),
        "native_completion": stats.get("native_tokens_completion"),
        "native_reasoning": stats.get("native_tokens_reasoning"),
        # OpenRouter-normalized: generation stats, falling back to the usage block
        "or_prompt": _first_not_none(stats.get("tokens_prompt"), usage.get("prompt_tokens")),
        "or_completion": _first_not_none(stats.get("tokens_completion"), usage.get("completion_tokens")),
    }


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "N/A" if value is None else format(value, spec)


def print_verbose(result: BenchmarkResult) -> None:
    bar = "-" * 85
    print(f"\n{bar}")
    print(f"[{result.dataset} #{result.index}] FULL RESPONSE")
    print(bar)
    print(f"THINKING ({result.native_reasoning_tokens} native tokens, {result.reasoning_chars} chars):")
    print(result.reasoning if result.reasoning else "<no reasoning text returned>")
    print(f"\nANSWER ({result.native_answer_tokens} native tokens, {result.answer_chars} chars):")
    print(result.answer if result.answer else "<empty answer>")
    print(bar)


def print_individual(result: BenchmarkResult) -> None:
    print(f"\n[{result.dataset} #{result.index}] Provider: {result.provider}")
    print(f"  Generation time:        {result.generation_time_ms} ms")
    print(f"  Server latency:         {result.server_latency_ms} ms")
    print(f"  Prompt tokens:          native={result.native_prompt_tokens}  openrouter={result.or_prompt_tokens}")
    print(
        f"  Completion tokens:      native={result.native_completion_tokens}  openrouter={result.or_completion_tokens}"
    )
    print(f"    of which reasoning:   native={result.native_reasoning_tokens}")
    print(
        f"  Completion chars:       {result.completion_chars} "
        f"(thinking={result.reasoning_chars}, answer={result.answer_chars})"
    )
    print(f"  Native TPS:             {_fmt(result.native_tps)} tokens/s")
    print(f"  OpenRouter TPS:         {_fmt(result.or_tps)} tokens/s")
    print(f"  Chars per second:       {_fmt(result.chars_per_second)} chars/s")
    print(f"  Cost:                   ${result.cost:.8f}")
    print(f"  Client elapsed:         {result.client_elapsed_seconds:.3f}s")


def _avg_med(values: list[float]) -> tuple[str, str]:
    if not values:
        return "N/A", "N/A"
    return f"{statistics.mean(values):.2f}", f"{statistics.median(values):.2f}"


def print_summary(results: list[BenchmarkResult], failures: dict[str, int]) -> None:
    by_dataset: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_dataset[r.dataset].append(r)

    groups = dict(by_dataset)
    if len(groups) > 1:
        groups["ALL"] = results

    def section(title: str, getter, unit: str) -> None:
        print(f"\n{title}")
        print(f"  {'dataset':<10}{'n':>5}{'average':>14}{'median':>14}  {unit}")
        for name, group in groups.items():
            values = [v for v in (getter(r) for r in group) if v is not None]
            avg, med = _avg_med(values)
            print(f"  {name:<10}{len(values):>5}{avg:>14}{med:>14}")

    print("\n" + "=" * 85)
    print("SUMMARY")
    print("=" * 85)

    section("1. Request time (client-side, end-to-end)", lambda r: r.client_elapsed_seconds, "[seconds]")
    section("2. Native tokens per second", lambda r: r.native_tps, "[native tokens/s]")
    section("3. OpenRouter tokens per second", lambda r: r.or_tps, "[OR tokens/s]")
    section("4. Chars per second (thinking + answer)", lambda r: r.chars_per_second, "[chars/s]")
    section("5. Native completion tokens (thinking + answer)", lambda r: r.native_completion_tokens, "[tokens]")
    section("6. Completion chars (thinking + answer)", lambda r: r.completion_chars, "[chars]")

    total_cost = sum(r.cost for r in results)
    total_prompt_native = sum(r.native_prompt_tokens or 0 for r in results)
    total_prompt_or = sum(r.or_prompt_tokens or 0 for r in results)
    print(f"\nSuccessful requests: {len(results)}")
    if failures:
        print(f"Failed requests: {dict(failures)}")
    print(f"Total prompt tokens: native={total_prompt_native}  openrouter={total_prompt_or}")
    print(f"Total cost: ${total_cost:.8f}")


def build_result(
    pending: PendingGeneration,
    stats: dict | None,
    or_config: OpenRouterConfig,
) -> BenchmarkResult:
    message = pending.completion["choices"][0]["message"]
    usage = pending.completion.get("usage") or {}
    counts = extract_token_counts(stats, usage)
    stats = stats or {}

    return BenchmarkResult(
        dataset=pending.dataset,
        index=pending.index,
        answer=message.get("content") or "",
        reasoning=message.get("reasoning") or "",
        provider=stats.get("provider_name", or_config.inference_provider),
        generation_time_ms=stats.get("generation_time"),
        server_latency_ms=stats.get("latency"),
        native_prompt_tokens=counts["native_prompt"],
        native_completion_tokens=counts["native_completion"],
        native_reasoning_tokens=counts["native_reasoning"],
        or_prompt_tokens=counts["or_prompt"],
        or_completion_tokens=counts["or_completion"],
        cost=stats.get("total_cost", usage.get("cost", 0.0)) or 0.0,
        client_elapsed_seconds=pending.client_elapsed,
        model_id=or_config.model_id,
        thinking_effort=or_config.thinking_effort,
    )


def run_combo(
    client: httpx.Client,
    or_config: OpenRouterConfig,
    prompts: dict[str, list[str]],
    cfg: DictConfig,
) -> tuple[list[BenchmarkResult], dict[str, int]]:
    failures: dict[str, int] = defaultdict(int)

    # ---- Phase 1: all generations, one after the other ----
    pending: list[PendingGeneration] = []
    for dataset, dataset_prompts in prompts.items():
        for index, prompt in enumerate(dataset_prompts):
            print(f"\n[{dataset}] Generation {index + 1}/{len(dataset_prompts)}")
            try:
                completion, elapsed = send_request(client, prompt, or_config)
            except httpx.HTTPError as exc:
                failures[dataset] += 1
                print(f"  FAILED: {exc!r}")
                continue
            pending.append(PendingGeneration(dataset, index, completion, elapsed))
            print(f"  Done in {elapsed:.2f}s (id={completion['id']})")

    # ---- Phase 2: wait once, then fetch stats for all of them ----
    print(f"\nWaiting {cfg.stats_initial_wait}s before fetching stats for {len(pending)} generations...")
    time.sleep(cfg.stats_initial_wait)

    results: list[BenchmarkResult] = []
    for n, p in enumerate(pending, 1):
        print(f"\n[{p.dataset}] Fetching stats {n}/{len(pending)}")
        try:
            stats = get_generation_stats(
                client,
                p.completion["id"],
                max_attempts=cfg.stats_max_attempts,
                initial_delay=cfg.stats_retry_delay,
            )
        except httpx.HTTPError as exc:
            print(f"  STATS FAILED: {exc!r}")
            stats = None
        if stats is None:
            print("  WARNING: no generation stats; falling back to usage block (TPS will be N/A)")

        result = build_result(p, stats, or_config)
        results.append(result)
        if or_config.verbose:
            print_verbose(result)

    # ---- Report: all stats at the end ----
    print("\n" + "=" * 85)
    print(f"INDIVIDUAL RESULTS: {or_config.model_id} | thinking={or_config.thinking_effort}")
    print("=" * 85)
    for r in results:
        print_individual(r)
    if results:
        print_summary(results, failures)
    else:
        print("\nNo successful requests.")

    return results, failures


def load_config() -> DictConfig:
    parser = argparse.ArgumentParser(description="Benchmark OpenRouter inference speed.")
    parser.add_argument("--config", type=Path, default=Path("configs/or_config.yaml"))
    args, overrides = parser.parse_known_args()  # extra args are dotlist overrides: key=value

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    if not cfg.api_key:
        raise SystemExit("Set OPENROUTER_API_KEY or provide api_key=... ")
    if cfg.len_subset < 1:
        raise SystemExit("len_subset must be positive.")
    return cfg


def save_json(cfg: DictConfig, runs: list[dict], rows: list[dict]) -> None:
    public_cfg = OmegaConf.to_container(cfg, resolve=True)
    public_cfg.pop("api_key", None)  # never write the key to disk

    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": public_cfg, "runs": runs, "results": rows}, indent=2))


def main() -> None:
    cfg = load_config()

    combos = [(model, effort) for model in cfg.models for effort in model.get("thinking_efforts", cfg.thinking_efforts)]
    print(f"Planned runs: {len(combos)}")
    for model, effort in combos:
        ds = list(model.get("datasets", cfg.datasets))
        print(f"  {model.model_id} | thinking={effort} | datasets={ds}")

    prompt_cache: dict[tuple, dict[str, list[str]]] = {}
    runs: list[dict] = []
    rows: list[dict] = []

    with create_client(api_key=cfg.api_key) as client:
        for run_no, (model, effort) in enumerate(combos, 1):
            datasets = tuple(model.get("datasets", cfg.datasets))
            if datasets not in prompt_cache:
                prompt_cache[datasets] = build_prompts(
                    datasets=list(datasets),
                    len_subset=cfg.len_subset,
                    offset=cfg.offset,
                    data_dir=Path(cfg.data_dir),
                    random_seed=cfg.random_seed,
                )
            prompts = prompt_cache[datasets]

            or_config = OpenRouterConfig(
                api_key=cfg.api_key,
                model_id=model.model_id,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                thinking_effort=effort,
                inference_provider=model.inference_provider,
                random_seed=cfg.random_seed,
                verbose=cfg.verbose,
            )

            print("\n" + "#" * 85)
            print(f"# RUN {run_no}/{len(combos)}: {or_config.model_id} | thinking={effort} | datasets={list(datasets)}")
            print("#" * 85)

            results, failures = run_combo(client, or_config, prompts, cfg)

            runs.append(
                {
                    "model_id": or_config.model_id,
                    "thinking_effort": effort,
                    "provider": or_config.inference_provider,
                    "datasets": list(datasets),
                    "n_success": len(results),
                    "failures": dict(failures),
                    "total_cost": sum(r.cost for r in results),
                }
            )
            rows.extend(r.to_row() for r in results)

            # save after every run so a crash in a long job doesn't lose everything
            save_json(cfg, runs, rows)

    print(f"\nAll runs finished. Results table written to {cfg.output}")


if __name__ == "__main__":
    main()
