"""Send the prepared questions to an LLM API and report time / token usage per dataset.

Environment:
    LLM_BASE_URL   e.g. https://api.example.com/v1  (or the full .../chat/completions URL)
    LLM_API_KEY

Two thinking-control styles are supported (--api-style):
    openai  : OpenAI-compatible, `reasoning_effort` = none|low|medium|high|max
    budget  : special API, `"thinking": {"type": "enabled", "budget_tokens": N}` is added to the
              payload, or removed entirely for --thinking none

Token provenance (every token count is stored together with its source):
    api        reported verbatim by the API
    derived    exact arithmetic on API-reported numbers (e.g. answer = completion - thinking)
    estimated  chars/4 heuristic, NOT exact
    assumed    value inferred from a hypothesis (e.g. "no thinking requested -> 0 thinking")
    missing    not available
"""

import argparse
import asyncio
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
from rich.console import Console
from rich.table import Table

from llm_verbosity.filter_data import DATASETS

log = logging.getLogger("llm_verbosity")
console = Console()

SYSTEM_PROMPT = (
    "You are a careful expert problem solver. Solve the question and finish your reply "
    "with a final line of the form 'Answer: <your final answer>'."
)

THINKING_CHOICES = ("none", "low", "medium", "high", "max")
THINKING_BUDGETS = {"low": 2048, "medium": 8192, "high": 16384, "max": 24576}  # --api-style budget
RETRY_STATUS = {408, 409, 425, 429}
TRUNCATED_REASONS = {"length", "max_tokens", "max_output_tokens"}

# --- token provenance
API, DERIVED, ESTIMATED, ASSUMED, MISSING = "api", "derived", "estimated", "assumed", "missing"
SOURCES = (API, DERIVED, ESTIMATED, ASSUMED, MISSING)
EXACT_SOURCES = {API, DERIVED}

TOKEN_METRICS = (
    "prompt_tokens",
    "thinking_tokens",
    "answer_tokens",  # tokens produced AFTER thinking (visible answer only)
    "completion_tokens",  # everything the model generated: thinking + answer
    "total_tokens",  # prompt + completion
)
KEY_METRICS = ("thinking_tokens", "answer_tokens", "completion_tokens", "total_tokens")
STAT_METRICS = ("elapsed_s", *TOKEN_METRICS)

WARNING_DESCRIPTIONS = {
    "no_usage": "API returned no usage block -> all token counts estimated from text length",
    "estimated_tokens": "some token counts are ESTIMATED (chars/4), not reported by the API",
    "assumed_tokens": "some token counts rest on an assumption (see JSON *_source fields)",
    "hidden_thinking_suspected": "thinking requested but no thinking tokens/text observed",
    "thinking_exceeds_completion": "reported thinking tokens > completion tokens",
    "answer_mismatch": "answer_tokens disagree strongly with the visible answer length",
    "missing_prompt_tokens": "prompt_tokens not reported -> total_tokens unavailable",
    "truncated": "generation stopped at the max token limit",
    "empty_answer": "no visible answer text (budget probably spent on thinking)",
    "retried": "request needed more than one attempt (time = successful attempt only)",
}


# --------------------------------------------------------------------------- errors


class ApiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class EmptyResponseError(Exception):
    pass


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ApiError):
        return exc.status in RETRY_STATUS or exc.status >= 500
    return isinstance(exc, httpx.TransportError | json.JSONDecodeError | EmptyResponseError)


# --------------------------------------------------------------------------- request


def format_message(question: dict, model_id: str) -> list[dict]:
    if question["image"]:  # "" if not multi-modal
        content = [
            {"type": "text", "text": question["question"]},
            {"type": "image_url", "image_url": {"url": question["image"]}},
        ]
    else:
        content = question["question"]

    system_role = "user" if "o1" in model_id else "system"  # o1: no system prompt
    return [
        {"role": system_role, "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_payload(args: argparse.Namespace, messages: list[dict]) -> dict:
    payload: dict = {
        "model": args.model_id,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": True,
        "messages": messages,
    }
    if args.api_style == "openai":
        payload["max_completion_tokens"] = args.max_completion_tokens
        payload["stream_options"] = {"include_usage": True}
        payload["reasoning_effort"] = args.thinking
    else:  # budget
        payload["max_tokens"] = args.max_completion_tokens
        payload.update(web_tools=False, code_tools=False, canvas=False)
        if args.thinking != "none":
            budget = args.thinking_budget or THINKING_BUDGETS[args.thinking]
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return payload


def resolve_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith(("completions", "messages")):
        return url
    return f"{url}/chat/completions"


@dataclass
class StreamState:
    thinking: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None


def consume_event(event: dict, st: StreamState) -> None:
    """Handle both OpenAI-style chunks and `content_block_delta`-style events."""
    if err := event.get("error"):
        raise ApiError(500, json.dumps(err)[:500])

    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content"):
            st.text.append(delta["content"])
        if reasoning := (delta.get("reasoning_content") or delta.get("reasoning")):
            st.thinking.append(reasoning)
        if choice.get("finish_reason"):
            st.finish_reason = choice["finish_reason"]

    etype = event.get("type")
    delta = event.get("delta") or {}
    if etype == "content_block_delta":
        if delta.get("type") == "text_delta":
            st.text.append(delta.get("text", ""))
        elif delta.get("type") == "thinking_delta":
            st.thinking.append(delta.get("thinking", ""))
    elif etype == "message_delta" and delta.get("stop_reason"):
        st.finish_reason = delta["stop_reason"]

    for u in (event.get("usage"), (event.get("message") or {}).get("usage")):
        if isinstance(u, dict):
            st.usage.update({k: v for k, v in u.items() if v is not None})


async def stream_completion(client: httpx.AsyncClient, url: str, headers: dict, payload: dict) -> StreamState:
    st = StreamState()
    async with client.stream("POST", url, json=payload, headers=headers) as r:
        if r.status_code >= 400:
            body = (await r.aread()).decode(errors="replace")[:500]
            raise ApiError(r.status_code, body)
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                break
            consume_event(json.loads(data), st)
    return st


async def request_with_retries(
    client: httpx.AsyncClient, url: str, headers: dict, payload: dict, retries: int, label: str
) -> tuple[StreamState, float, int]:
    """Returns (stream state, elapsed seconds of the successful attempt, attempts used)."""
    last: Exception | None = None
    for attempt in range(1, retries + 2):
        t0 = time.perf_counter()
        try:
            st = await stream_completion(client, url, headers, payload)
            elapsed = time.perf_counter() - t0
            if not st.thinking and not st.text:
                raise EmptyResponseError("empty response")
            return st, elapsed, attempt
        except Exception as exc:
            last = exc
            if not is_retryable(exc) or attempt > retries:
                break
            delay = min(60.0, 2.0**attempt) * (0.5 + random.random())
            log.warning(
                "%s attempt %d/%d failed (%r); retrying in %.1fs",
                label,
                attempt,
                retries + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    raise RuntimeError(f"giving up: {last!r}") from last


# --------------------------------------------------------------------------- token accounting

_estimate_warning_emitted = False


def _chars_over_4(s: str) -> int:
    """Raw heuristic, silent. Use est_tokens() when the value is actually REPORTED."""
    return math.ceil(len(s) / 4)


def est_tokens(s: str, what: str) -> int:
    """chars/4 fallback when the API does not report a count. Loud on first use."""
    global _estimate_warning_emitted
    if not _estimate_warning_emitted:
        _estimate_warning_emitted = True
        log.warning(
            "ESTIMATING TOKENS: the API did not report '%s' (first occurrence; further ones are "
            "counted in the final summary). chars/4 is a ROUGH heuristic that can be far off, "
            "especially for LaTeX, code and non-English text. Estimated values are marked "
            "source='estimated' in the JSON and with '~' in the table.",
            what,
        )
    else:
        log.debug("estimating tokens for '%s' with chars/4", what)
    return _chars_over_4(s)


def _first(d: dict, *keys: str) -> int | None:
    for k in keys:
        if isinstance(d.get(k), int | float):
            return int(d[k])
    return None


@dataclass
class TokenUsage:
    values: dict[str, int | None]
    sources: dict[str, str]
    warnings: list[tuple[str, str]]  # (code, message)


def parse_usage(usage: dict, thinking_text: str, text: str, thinking_requested: bool) -> TokenUsage:
    """Turn the raw `usage` block into token counts WITH provenance.

    answer_tokens is the clean "after thinking" measure:
      * thinking AND completion reported by the API  -> answer = completion - thinking (derived, exact)
      * API reports completion only, no thinking requested/observed -> answer = completion (assumed)
      * API reports completion only, thinking present -> answer = chars/4 of visible text
        (estimated) and thinking = completion - answer (estimated); the visible answer is
        used because streamed thinking may be summarized/hidden and would undercount.
      * nothing reported -> everything estimated from text length
    """
    w: list[tuple[str, str]] = []
    v: dict[str, int | None] = {}
    s: dict[str, str] = {}

    if not usage:
        w.append(("no_usage", "API returned no usage block; ALL token counts are estimated"))

    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    prompt = _first(usage, "prompt_tokens", "input_tokens")
    completion = _first(usage, "completion_tokens", "output_tokens")
    thinking = _first(details, "reasoning_tokens", "thinking_tokens")
    if thinking is None:
        thinking = _first(usage, "reasoning_tokens", "thinking_tokens")
    answer = _first(details, "answer_tokens", "text_tokens")  # rare, honoured if present
    total = _first(usage, "total_tokens")

    # ---- prompt
    v["prompt_tokens"], s["prompt_tokens"] = (prompt, API) if prompt is not None else (None, MISSING)

    # ---- completion / thinking / answer
    if completion is not None:
        v["completion_tokens"], s["completion_tokens"] = completion, API
        if thinking is not None and thinking > completion:
            w.append(
                (
                    "thinking_exceeds_completion",
                    f"thinking_tokens={thinking} > completion_tokens={completion}; assuming the API "
                    "reports completion WITHOUT thinking tokens, answer_tokens = completion_tokens",
                )
            )
            v["thinking_tokens"], s["thinking_tokens"] = thinking, API
            v["answer_tokens"], s["answer_tokens"] = (completion, ASSUMED) if answer is None else (answer, API)
        elif thinking is not None:
            v["thinking_tokens"], s["thinking_tokens"] = thinking, API
            if answer is not None:
                v["answer_tokens"], s["answer_tokens"] = answer, API
            else:
                v["answer_tokens"], s["answer_tokens"] = completion - thinking, DERIVED
        elif answer is not None:
            v["answer_tokens"], s["answer_tokens"] = answer, API
            v["thinking_tokens"], s["thinking_tokens"] = max(completion - answer, 0), DERIVED
        elif not thinking_requested and not thinking_text:
            v["thinking_tokens"], s["thinking_tokens"] = 0, ASSUMED
            v["answer_tokens"], s["answer_tokens"] = completion, ASSUMED
        else:
            if not thinking_text:
                w.append(
                    (
                        "hidden_thinking_suspected",
                        "thinking was requested but no thinking text was streamed and no thinking "
                        "tokens were reported; thinking is inferred as completion - est(answer)",
                    )
                )
            ans = est_tokens(text, "answer_tokens")
            v["answer_tokens"], s["answer_tokens"] = ans, ESTIMATED
            v["thinking_tokens"], s["thinking_tokens"] = max(completion - ans, 0), ESTIMATED
    else:
        # nothing about the completion reported: fall back to text length
        if thinking is not None:
            v["thinking_tokens"], s["thinking_tokens"] = thinking, API
        elif thinking_text:
            v["thinking_tokens"], s["thinking_tokens"] = (
                est_tokens(thinking_text, "thinking_tokens"),
                ESTIMATED,
            )
        else:
            v["thinking_tokens"], s["thinking_tokens"] = 0, ASSUMED
        if answer is not None:
            v["answer_tokens"], s["answer_tokens"] = answer, API
        else:
            v["answer_tokens"], s["answer_tokens"] = est_tokens(text, "answer_tokens"), ESTIMATED
        both_api = s["thinking_tokens"] == API and s["answer_tokens"] == API
        v["completion_tokens"] = (v["thinking_tokens"] or 0) + (v["answer_tokens"] or 0)
        s["completion_tokens"] = DERIVED if both_api else ESTIMATED

    # ---- sanity check: derived/api answer length vs. what we actually received as text
    if s["answer_tokens"] in (API, DERIVED, ASSUMED) and text:
        visible, ans = _chars_over_4(text), v["answer_tokens"] or 0
        if abs(ans - visible) > 100 and not (visible / 3 <= ans <= visible * 3):
            w.append(
                (
                    "answer_mismatch",
                    f"answer_tokens={ans} vs ~{visible} (chars/4 of the visible text); hidden "
                    "reasoning counted as answer, or an unusual tokenizer?",
                )
            )

    # ---- total
    c_src = s["completion_tokens"]
    if total is not None:
        v["total_tokens"], s["total_tokens"] = total, API
    elif prompt is not None:
        v["total_tokens"] = prompt + (v["completion_tokens"] or 0)
        s["total_tokens"] = DERIVED if c_src in EXACT_SOURCES else ESTIMATED
    else:
        v["total_tokens"], s["total_tokens"] = None, MISSING
        w.append(("missing_prompt_tokens", "prompt_tokens not reported -> total_tokens unavailable"))

    # ---- aggregated provenance warnings
    est = [m for m in KEY_METRICS if s.get(m) == ESTIMATED]
    if est:
        w.append(("estimated_tokens", f"estimated (chars/4): {', '.join(est)}"))
    ass = [m for m in KEY_METRICS if s.get(m) == ASSUMED]
    if ass:
        w.append(("assumed_tokens", f"assumed: {', '.join(ass)}"))
    return TokenUsage(v, s, w)


# --------------------------------------------------------------------------- jobs


@dataclass
class Result:
    tag: str
    index: int  # index inside the prepared parquet file
    id: str
    ok: bool = False
    elapsed_s: float | None = None  # successful attempt only
    wall_with_retries_s: float | None = None
    finish_reason: str | None = None
    attempts: int = 0
    tokens: dict = field(default_factory=dict)
    token_sources: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)  # warning codes
    error: str | None = None


def add_warning(res: Result, code: str, msg: str, level: int = logging.WARNING) -> None:
    res.warnings.append(code)
    log.log(level, "[%s] #%d %s: %s", res.tag, res.index, code, msg)


def _edges(s: str, n: int = 30) -> str:
    return f"first{n}={s[:n]!r} last{n}={s[-n:]!r}"


def log_verbose(tag: str, index: int, thinking: str, text: str) -> None:
    parts = [f"[{tag}] #{index}"]
    if thinking:
        parts.append(f"thinking: {_edges(thinking)}")
    parts.append(f"completion: {_edges(text)}")
    log.info(" | ".join(parts))


async def run_job(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    url: str,
    api_key: str,
    tag: str,
    index: int,
    row: dict,
) -> Result:
    res = Result(tag=tag, index=index, id=str(row["id"]))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Session-Id": f"verbosity-{tag}-{row['id']}",
    }
    payload = build_payload(args, format_message(row, args.model_id))
    t_start = time.perf_counter()
    try:
        st, elapsed, attempts = await request_with_retries(
            client, url, headers, payload, args.retries, f"[{tag}] #{index}"
        )
    except Exception as exc:
        res.error = repr(exc)
        log.error("[%s] #%d FAILED permanently: %s", tag, index, res.error)
        return res

    thinking_text, text = "".join(st.thinking), "".join(st.text)
    res.ok = True
    res.elapsed_s = round(elapsed, 3)
    res.wall_with_retries_s = round(time.perf_counter() - t_start, 3)
    res.attempts = attempts
    res.finish_reason = st.finish_reason

    usage = parse_usage(st.usage, thinking_text, text, args.thinking != "none")
    res.tokens, res.token_sources = usage.values, usage.sources
    for code, msg in usage.warnings:
        add_warning(res, code, msg)

    if st.finish_reason in TRUNCATED_REASONS:
        add_warning(
            res,
            "truncated",
            f"finish_reason={st.finish_reason!r}; raise --max-completion-tokens (token counts are capped by the limit)",
        )
    if not text.strip():
        add_warning(res, "empty_answer", f"no answer text (finish_reason={st.finish_reason!r})")
    if attempts > 1:
        add_warning(res, "retried", f"succeeded after {attempts} attempts", logging.INFO)

    if args.verbose:
        log_verbose(tag, index, thinking_text, text)
    return res


def select_indices(n_rows: int, args: argparse.Namespace, tag: str) -> list[int]:
    if args.limit and args.limit > n_rows - (0 if args.subset_seed is not None else args.offset):
        log.warning("[%s] --limit %d exceeds available questions; using fewer", tag, args.limit)
    if args.subset_seed is not None:  # random subset, reproducible and independent per dataset
        k = min(args.limit, n_rows)
        return sorted(random.Random(f"{args.subset_seed}-{tag}").sample(range(n_rows), k))
    end = args.offset + args.limit if args.limit else n_rows
    return list(range(n_rows))[args.offset : end]


async def run_all(args: argparse.Namespace, url: str, api_key: str) -> list[Result]:
    queue: asyncio.Queue = asyncio.Queue()
    for tag in args.datasets:
        path = args.data_dir / DATASETS[tag][0]
        if not path.exists():
            sys.exit(f"missing {path}; run `make download-data` first")
        rows = list(pl.read_parquet(path).iter_rows(named=True))
        for index in select_indices(len(rows), args, tag):
            queue.put_nowait((tag, index, rows[index]))

    total = queue.qsize()
    if total == 0:
        sys.exit("no questions selected (check --limit/--offset/--datasets)")
    log.info("running %d questions on %d workers", total, args.workers)
    results: list[Result] = []

    async def worker(client: httpx.AsyncClient) -> None:
        while True:
            try:
                tag, index, row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await run_job(client, args, url, api_key, tag, index, row))
            if not args.verbose:
                print(f"\rprogress: {len(results)}/{total}", end="", file=sys.stderr, flush=True)

    timeout = httpx.Timeout(args.timeout, connect=30.0)
    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await asyncio.gather(*(worker(client) for _ in range(args.workers)))
    if not args.verbose:
        print(file=sys.stderr)
    order = list(DATASETS)
    return sorted(results, key=lambda r: (order.index(r.tag), r.index))


# --------------------------------------------------------------------------- stats


def metric_value(r: Result, metric: str) -> float | None:
    return r.elapsed_s if metric == "elapsed_s" else r.tokens.get(metric)


def summarize(values: list[float]) -> dict:
    p90 = statistics.quantiles(values, n=10, method="inclusive")[-1] if len(values) > 1 else values[0]
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": p90,
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def build_summary(results: list[Result], tags: list[str]) -> dict:
    summary = {}
    for tag in tags:
        level = DATASETS[tag][1]
        rs = [r for r in results if r.tag == tag]
        ok = [r for r in rs if r.ok]
        metrics: dict[str, dict | None] = {}
        for m in STAT_METRICS:
            pairs = [(r, metric_value(r, m)) for r in ok]
            pairs = [(r, x) for r, x in pairs if x is not None]
            if not pairs:
                metrics[m] = None
                continue
            stat = summarize([x for _, x in pairs])
            if m != "elapsed_s":
                src = Counter(r.token_sources.get(m, MISSING) for r, _ in pairs)
                stat["sources"] = {k: src.get(k, 0) for k in SOURCES}
                stat["inexact"] = sum(c for k, c in src.items() if k not in EXACT_SOURCES)
            metrics[m] = stat
        summary[tag] = {
            "difficulty": level,
            "n": len(rs),
            "ok": len(ok),
            "failed": len(rs) - len(ok),
            "all_key_metrics_exact": sum(all(r.token_sources.get(m) in EXACT_SOURCES for m in KEY_METRICS) for r in ok),
            "warning_counts": dict(Counter(c for r in rs for c in r.warnings)),
            "indices": [r.index for r in rs],
            "metrics": metrics,
        }
    return summary


def print_table(summary: dict, args: argparse.Namespace) -> None:
    table = Table(
        title=f"{args.model_id} | top_p={args.top_p} temp={args.temperature} thinking={args.thinking}",
        caption=(
            "cells: mean (median) | '~' = at least one value is estimated/assumed, not API-exact | "
            "Answer = tokens after thinking | Completion = Thinking + Answer"
        ),
    )
    cols = [
        ("Dataset", "left"),
        ("Level", "left"),
        ("OK/N", "right"),
        ("Time (s)", "right"),
        ("Thinking tok", "right"),
        ("Answer tok (post-thinking)", "right"),
        ("Completion tok", "right"),
        ("Total tok", "right"),
        ("API-exact", "right"),
    ]
    for name, just in cols:
        table.add_column(name, justify=just)

    def cell(stat: dict | None, digits: int) -> str:
        if not stat:
            return "n/a"
        mark = "~" if stat.get("inexact") else ""
        return f"{mark}{stat['mean']:.{digits}f} ({stat['median']:.{digits}f})"

    for tag, s in summary.items():
        m = s["metrics"]
        table.add_row(
            tag,
            s["difficulty"],
            f"{s['ok']}/{s['n']}",
            cell(m["elapsed_s"], 1),
            cell(m["thinking_tokens"], 0),
            cell(m["answer_tokens"], 0),
            cell(m["completion_tokens"], 0),
            cell(m["total_tokens"], 0),
            f"{s['all_key_metrics_exact']}/{s['ok']}",
        )
    console.print(table)

    prov = Table(title="Token provenance (number of questions per source)")
    prov.add_column("Dataset")
    prov.add_column("Metric")
    for src in SOURCES:
        prov.add_column(src, justify="right")
    for tag, s in summary.items():
        for metric in KEY_METRICS:
            stat = s["metrics"][metric]
            counts = stat["sources"] if stat else dict.fromkeys(SOURCES, 0)
            prov.add_row(tag, metric, *(str(counts[k]) if counts[k] else "-" for k in SOURCES))
    console.print(prov)


def print_warnings(results: list[Result], args: argparse.Namespace) -> None:
    codes = Counter(c for r in results for c in r.warnings)
    failed = [r for r in results if not r.ok]
    lines: list[str] = []
    if failed:
        lines.append(f"{len(failed)} request(s) FAILED permanently and are excluded from the stats:")
        lines += [f"    [{r.tag}] #{r.index}: {r.error}" for r in failed[:10]]
        if len(failed) > 10:
            lines.append(f"    ... and {len(failed) - 10} more (see JSON)")
    for code, n in codes.most_common():
        lines.append(f"{n:>4} x {code}: {WARNING_DESCRIPTIONS.get(code, '')}")

    ok = [r for r in results if r.ok]
    think = [r.tokens.get("thinking_tokens") or 0 for r in ok]
    if ok and args.thinking != "none" and not any(think):
        lines.append(
            f"thinking={args.thinking} was requested but ZERO thinking tokens were observed in "
            f"every question: the endpoint/model probably ignores '{args.api_style}' thinking control"
        )
    if ok and args.thinking == "none" and sum(1 for t in think if t) > len(ok) / 2:
        lines.append(
            "thinking=none but most questions still used thinking tokens: the model may think regardless of the request"
        )
    if lines:
        console.print("\n[bold yellow]WARNINGS[/bold yellow]")
        for line in lines:
            console.print(f"[yellow]- {line}[/yellow]", markup=True, highlight=False)
    else:
        console.print("\n[green]No warnings.[/green]")


def output_path(args: argparse.Namespace) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model_id)
    return args.output_dir / f"{safe_model}_{args.top_p}_{args.temperature}_thinking_{args.thinking}.json"


# --------------------------------------------------------------------------- cli


def thinking_type(value: str) -> str:
    v = value.lower()
    v = "medium" if v == "mid" else v
    if v not in THINKING_CHOICES:
        raise argparse.ArgumentTypeError(f"choose from {THINKING_CHOICES} (or 'mid')")
    return v


def dataset_type(value: str) -> str:
    v = value.lower()
    for tag, (_, level) in DATASETS.items():
        if v in (tag.lower(), level):  # accept "MATH-500" or "easy", etc.
            return tag
    raise argparse.ArgumentTypeError(f"unknown dataset {value!r}; use one of {list(DATASETS)} or easy|mid|hard")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--top-p", type=float, required=True)
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--thinking", type=thinking_type, default="none")
    p.add_argument("--api-style", choices=("openai", "budget"), default="openai")
    p.add_argument("--thinking-budget", type=int, default=None, help="override budget_tokens")
    p.add_argument("--max-completion-tokens", type=int, default=32768)
    p.add_argument("--workers", "-n", type=int, default=8)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--timeout", type=float, default=600.0)

    sub = p.add_argument_group("subset selection")
    sub.add_argument("--limit", type=int, default=0, help="questions per dataset (0 = all)")
    sub.add_argument("--offset", type=int, default=0, help="skip the first K questions")
    sub.add_argument(
        "--subset-seed",
        type=int,
        default=None,
        help="pick --limit questions at random (reproducible) instead of the first ones",
    )
    sub.add_argument(
        "--datasets",
        nargs="+",
        type=dataset_type,
        default=list(DATASETS),
        help="subset of datasets: MATH-500 GPQA-Diamond HLE-Diamond (or easy mid hard)",
    )

    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("llm_verbosity_stats"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.limit < 0 or args.offset < 0:
        p.error("--limit and --offset must be >= 0")
    if args.subset_seed is not None and not args.limit:
        p.error("--subset-seed needs --limit")
    if args.subset_seed is not None and args.offset:
        p.error("--subset-seed and --offset are mutually exclusive")
    args.datasets = list(dict.fromkeys(args.datasets))  # dedupe, keep order
    return args


def main() -> None:
    args = parse_args()
    base_url, api_key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
    if not base_url or not api_key:
        sys.exit("LLM_BASE_URL and LLM_API_KEY must be set in the environment")

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    url = resolve_url(base_url)
    results = asyncio.run(run_all(args, url, api_key))
    summary = build_summary(results, args.datasets)
    print_table(summary, args)
    print_warnings(results, args)

    path = output_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(base_url=url, timestamp=datetime.now(UTC).isoformat())
    path.write_text(
        json.dumps(
            {"config": config, "summary": summary, "results": [asdict(r) for r in results]},
            indent=2,
        )
    )
    print(f"saved stats -> {path}")


if __name__ == "__main__":
    main()
