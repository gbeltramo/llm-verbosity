import argparse
import json
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
from omegaconf import DictConfig, OmegaConf

from .datasets import build_prompts


@dataclass
class NaiveResult:
    dataset: str
    index: int
    model_id: str
    thinking: bool
    answer: str = ""
    reasoning: str = ""
    # tokens (server-reported)
    total_tokens: int | None = None  # thinking + answer (usage.output_tokens)
    thinking_tokens: int | None = None
    answer_tokens: int | None = None
    prompt_tokens: int | None = None  # prompt_n (non-cached)
    cached_prompt_tokens: int | None = None
    # speed
    server_tps: float | None = None  # naivert predicted_n / predicted_ms
    device_tps: float | None = None  # naivert decode_tok_s_device
    client_tps: float | None = None  # total_tokens / (end - first token)
    ttft_seconds: float | None = None
    client_elapsed_seconds: float = 0.0
    reasoning_chars: int = 0
    answer_chars: int = 0
    timings: list[dict] = field(default_factory=list)  # raw naivert timings (one per message_delta)

    def to_row(self) -> dict:
        return asdict(self)


def create_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(600.0, connect=20.0))


def build_payload(prompt: str, model_id: str, thinking: bool, cfg: DictConfig, session_id: str) -> dict:
    payload = {
        "model": model_id,
        "max_tokens": cfg.max_tokens,
        "stream": True,  # always streaming
        "seed": cfg.random_seed,
        "metadata": {"user_id": session_id},
        "web_tools": cfg.web_tools,
        "code_tools": cfg.code_tools,
        "canvas": cfg.canvas,
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking:  # thinking is on only when this key is present
        payload["thinking"] = {"type": "enabled", "budget_tokens": cfg.thinking_budget}
    return payload


def send_request(client: httpx.Client, prompt: str, model_id: str, thinking: bool, cfg: DictConfig) -> NaiveResult:
    """Stream one request, parse the SSE events and collect stats."""
    session_id = f"bench-{uuid.uuid4().hex[:12]}"  # unique per request -> no cross-request session effects
    url = f"{cfg.base_url.rstrip('/')}/v1/agent"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
        "X-Session-Id": session_id,
    }
    payload = build_payload(prompt, model_id, thinking, cfg, session_id)

    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    usages: list[dict] = []
    timings: list[dict] = []
    first_token_t: float | None = None

    start = time.perf_counter()
    with client.stream("POST", url, json=payload, headers=headers) as response:
        if response.status_code >= 400:
            response.read()
            raise httpx.HTTPStatusError(
                f"{response.status_code}: {response.text[:500]}", request=response.request, response=response
            )
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data:
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")
            if etype == "content_block_delta":
                delta = event.get("delta", {})
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                if delta.get("type") == "thinking_delta":
                    reasoning_parts.append(delta.get("thinking", ""))
                elif delta.get("type") == "text_delta":
                    answer_parts.append(delta.get("text", ""))
            elif etype == "message_delta":
                usages.append(event.get("usage") or {})
                t = (event.get("naivert") or {}).get("timings")
                if t:
                    timings.append(t)
            elif etype == "error":
                raise RuntimeError(f"stream error: {event}")
    end = time.perf_counter()

    # Aggregate across message_delta events (more than one if the agent takes several steps)
    total_tokens = sum(u.get("output_tokens", 0) for u in usages) if usages else None
    thinking_tokens = (
        sum((u.get("output_tokens_details") or {}).get("thinking_tokens", 0) for u in usages) if usages else None
    )
    answer_tokens = total_tokens - thinking_tokens if total_tokens is not None else None

    pred_n = sum(t.get("predicted_n", 0) for t in timings)
    pred_ms = sum(t.get("predicted_ms", 0.0) for t in timings)
    server_tps = pred_n / pred_ms * 1000 if pred_ms > 0 else None

    device_vals = [t["decode_tok_s_device"] for t in timings if t.get("decode_tok_s_device") is not None]
    device_tps = statistics.mean(device_vals) if device_vals else None

    client_tps = None
    if first_token_t is not None and total_tokens and end > first_token_t:
        client_tps = total_tokens / (end - first_token_t)

    reasoning = "".join(reasoning_parts)
    answer = "".join(answer_parts)

    return NaiveResult(
        dataset="",
        index=-1,
        model_id=model_id,
        thinking=thinking,
        answer=answer,
        reasoning=reasoning,
        total_tokens=total_tokens,
        thinking_tokens=thinking_tokens,
        answer_tokens=answer_tokens,
        prompt_tokens=sum(t.get("prompt_n", 0) for t in timings) if timings else None,
        cached_prompt_tokens=sum(t.get("n_prompt_tokens_cache", 0) for t in timings) if timings else None,
        server_tps=server_tps,
        device_tps=device_tps,
        client_tps=client_tps,
        ttft_seconds=(first_token_t - start) if first_token_t else None,
        client_elapsed_seconds=end - start,
        reasoning_chars=len(reasoning),
        answer_chars=len(answer),
        timings=timings,
    )


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "N/A" if value is None else format(value, spec)


def print_verbose(r: NaiveResult) -> None:
    bar = "-" * 85
    print(f"\n{bar}\n[{r.dataset} #{r.index}] FULL RESPONSE\n{bar}")
    print(f"THINKING ({r.thinking_tokens} tokens, {r.reasoning_chars} chars):")
    print(r.reasoning or "<no reasoning text returned>")
    print(f"\nANSWER ({r.answer_tokens} tokens, {r.answer_chars} chars):")
    print(r.answer or "<empty answer>")
    print(bar)


def print_individual(r: NaiveResult) -> None:
    print(f"\n[{r.dataset} #{r.index}] model={r.model_id} thinking={r.thinking}")
    print(f"  Total tokens (think+answer): {_fmt(r.total_tokens, 'd')}")
    print(f"    thinking tokens:           {_fmt(r.thinking_tokens, 'd')}")
    print(f"    answer tokens:             {_fmt(r.answer_tokens, 'd')}")
    print(f"  Prompt tokens:               {_fmt(r.prompt_tokens, 'd')} (cached={_fmt(r.cached_prompt_tokens, 'd')})")
    print(f"  Server TPS (predicted):      {_fmt(r.server_tps)} tokens/s")
    print(f"  Device decode TPS:           {_fmt(r.device_tps)} tokens/s")
    print(f"  Client TPS (after 1st tok):  {_fmt(r.client_tps)} tokens/s")
    print(f"  TTFT:                        {_fmt(r.ttft_seconds, '.3f')}s")
    print(f"  Client elapsed:              {r.client_elapsed_seconds:.3f}s")


def _avg_med(values: list[float]) -> tuple[str, str]:
    if not values:
        return "N/A", "N/A"
    return f"{statistics.mean(values):.2f}", f"{statistics.median(values):.2f}"


def print_summary(results: list[NaiveResult], failures: dict[str, int]) -> None:
    by_dataset: dict[str, list[NaiveResult]] = defaultdict(list)
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
    section("1. Total generated tokens (thinking + answer)", lambda r: r.total_tokens, "[tokens]")
    section("2. Server tokens per second", lambda r: r.server_tps, "[tokens/s]")
    section("3. Device decode tokens per second", lambda r: r.device_tps, "[tokens/s]")
    section("4. Client tokens per second", lambda r: r.client_tps, "[tokens/s]")
    section("5. Thinking tokens", lambda r: r.thinking_tokens, "[tokens]")
    section("6. Request time (client-side, end-to-end)", lambda r: r.client_elapsed_seconds, "[seconds]")

    print(f"\nSuccessful requests: {len(results)}")
    if failures:
        print(f"Failed requests: {dict(failures)}")


def run_combo(
    client: httpx.Client,
    model_id: str,
    thinking: bool,
    prompts: dict[str, list[str]],
    cfg: DictConfig,
) -> tuple[list[NaiveResult], dict[str, int]]:
    failures: dict[str, int] = defaultdict(int)
    results: list[NaiveResult] = []

    for dataset, dataset_prompts in prompts.items():
        for index, prompt in enumerate(dataset_prompts):
            print(f"\n[{dataset}] Generation {index + 1}/{len(dataset_prompts)}")
            try:
                result = send_request(client, prompt, model_id, thinking, cfg)
            except (httpx.HTTPError, RuntimeError) as exc:
                failures[dataset] += 1
                print(f"  FAILED: {exc!r}")
                continue
            result.dataset = dataset
            result.index = index
            results.append(result)
            print(
                f"  Done in {result.client_elapsed_seconds:.2f}s | "
                f"tokens={result.total_tokens} | server_tps={_fmt(result.server_tps)}"
            )
            if cfg.verbose:
                print_verbose(result)

    print("\n" + "=" * 85)
    print(f"INDIVIDUAL RESULTS: {model_id} | thinking={thinking}")
    print("=" * 85)
    for r in results:
        print_individual(r)
    if results:
        print_summary(results, failures)
    else:
        print("\nNo successful requests.")
    return results, failures


def load_config() -> DictConfig:
    parser = argparse.ArgumentParser(description="Benchmark Naive endpoint inference speed.")
    parser.add_argument("--config", type=Path, default=Path("configs/naive_config.yaml"))
    args, overrides = parser.parse_known_args()  # extra args are dotlist overrides: key=value

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    if not cfg.base_url:
        raise SystemExit("Set NAIVE_BASE_URL or provide base_url=...")
    if not cfg.api_key:
        raise SystemExit("Set NAIVE_API_KEY or provide api_key=...")
    if cfg.len_subset < 1:
        raise SystemExit("len_subset must be positive.")
    if cfg.max_tokens <= cfg.thinking_budget:
        raise SystemExit("max_tokens must be larger than thinking_budget.")
    return cfg


def save_json(cfg: DictConfig, runs: list[dict], rows: list[dict]) -> None:
    public_cfg = OmegaConf.to_container(cfg, resolve=True)
    public_cfg.pop("api_key", None)  # never write the key to disk

    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": public_cfg, "runs": runs, "results": rows}, indent=2))


def main() -> None:
    cfg = load_config()

    combos = [
        (model, bool(thinking)) for model in cfg.models for thinking in model.get("thinking_modes", cfg.thinking_modes)
    ]
    print(f"Planned runs: {len(combos)}")
    for model, thinking in combos:
        ds = list(model.get("datasets", cfg.datasets))
        print(f"  {model.model_id} | thinking={thinking} | datasets={ds}")

    prompt_cache: dict[tuple, dict[str, list[str]]] = {}
    runs: list[dict] = []
    rows: list[dict] = []

    with create_client() as client:
        for run_no, (model, thinking) in enumerate(combos, 1):
            datasets = tuple(model.get("datasets", cfg.datasets))
            if datasets not in prompt_cache:
                prompt_cache[datasets] = build_prompts(
                    datasets=list(datasets),
                    len_subset=cfg.len_subset,
                    offset=cfg.offset,
                    data_dir=Path(cfg.data_dir),
                    random_seed=cfg.dataset_seed,
                )
            prompts = prompt_cache[datasets]

            print("\n" + "#" * 85)
            print(f"# RUN {run_no}/{len(combos)}: {model.model_id} | thinking={thinking} | datasets={list(datasets)}")
            print("#" * 85)

            results, failures = run_combo(client, model.model_id, thinking, prompts, cfg)

            runs.append(
                {
                    "model_id": model.model_id,
                    "thinking": thinking,
                    "thinking_budget": cfg.thinking_budget if thinking else None,
                    "datasets": list(datasets),
                    "n_success": len(results),
                    "failures": dict(failures),
                    "mean_total_tokens": (
                        statistics.mean(r.total_tokens for r in results if r.total_tokens is not None)
                        if any(r.total_tokens is not None for r in results)
                        else None
                    ),
                    "mean_server_tps": (
                        statistics.mean(r.server_tps for r in results if r.server_tps is not None)
                        if any(r.server_tps is not None for r in results)
                        else None
                    ),
                }
            )
            rows.extend(r.to_row() for r in results)

            # save after every run so a crash in a long job doesn't lose everything
            save_json(cfg, runs, rows)

    print(f"\nAll runs finished. Results table written to {cfg.output}")


if __name__ == "__main__":
    main()
