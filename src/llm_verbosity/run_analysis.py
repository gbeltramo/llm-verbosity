"""Send the prepared questions to an LLM API and report time / speed / completion tokens.

Environment:
    LLM_API_KEY    API key (an OpenRouter key for --api-style openrouter)
    LLM_BASE_URL   only for --api-style budget, e.g. https://api.example.com/v1

Two API styles (--api-style):
    openrouter : POSTs to https://openrouter.ai/api/v1/chat/completions (fixed).
                 "usage": {"include": true} asks for per-request token counts and cost,
                 "provider": {"only": [...], "allow_fallbacks": false} pins the provider and
                 "reasoning": {"enabled": ..., "effort": ...} controls thinking.
    budget     : special (agent) API, POSTed to {LLM_BASE_URL}/agent, with
                 "thinking": {"type": "enabled", "budget_tokens": N} when thinking is on.

Responses are read exactly as the endpoint returns them: a plain JSON body (OpenRouter) or an SSE
stream (the agent API streams even when the payload says "stream": false). The numbers always come
from the API's own response:
    usage.output_tokens / usage.completion_tokens     all generated tokens (thinking + answer)
    usage.output_tokens_details.thinking_tokens       the thinking share
    usage.completion_tokens_details.reasoning_tokens  (OpenRouter name for the same thing)
    naivert.timings of the last message_delta         engine decode speed (agent API only)
    agent_done.messages                               final thinking / answer text

Per request we report:
    elapsed_s          client-measured end-to-end request time
    tokens_per_s       generated tokens per second: the engine decode speed when the endpoint
                       reports one (agent API), otherwise generated tokens / elapsed_s
    completion_tokens  all tokens generated (thinking + answer), as reported by the API
"""

import argparse
import asyncio
import json
import logging
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

# --- defaults / constants
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PROVIDER = "wafer/us"
DEFAULT_TOP_P = 0.95
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_COMPLETION_TOKENS = 256_000
DEFAULT_THINKING_BUDGET = 4096  # budget style: budget_tokens is REQUIRED when thinking is on

API_STYLES = ("openrouter", "budget")
BUDGET_ENDPOINT = "agent"
ENDPOINT_RE = re.compile(r"/(agent|chat/completions|completions|messages)$")

THINKING_CHOICES = ("none", "low", "medium", "high", "max")
THINKING_ALIASES = {"mid": "medium", "off": "none"}

RETRY_STATUS = {408, 409, 425, 429}
TRUNCATED_REASONS = {"length", "max_tokens", "max_output_tokens"}
O1_RE = re.compile(r"(^|/)o1([-:]|$)")  # o1 family: no system prompt

TOKEN_METRICS = ("prompt_tokens", "thinking_tokens", "answer_tokens", "completion_tokens", "total_tokens")
STAT_METRICS = ("elapsed_s", "tokens_per_s", "e2e_tokens_per_s", *TOKEN_METRICS)
# engine timings worth keeping in the JSON (everything else is prefill debug data)
TIMING_KEYS = (
    "queue_wait_ms",
    "initialize_ms",
    "capture_ms",
    "prompt_n",
    "prompt_ms",
    "prompt_per_second",
    "n_prompt_tokens_cache",
    "predicted_n",
    "predicted_ms",
    "predicted_per_second",
    "decode_tok_s_device",
    "device_steps",
    "segments",
)

WARNING_DESCRIPTIONS = {
    "no_usage": "the response carried no usage block",
    "missing_tokens": "some token counts were not reported (shown as n/a)",
    "truncated": "generation hit the max token limit",
    "empty_answer": "no visible answer text (budget probably spent on thinking)",
    "retried": "request needed more than one attempt (time = successful attempt only)",
}


# --------------------------------------------------------------------------- errors


class ApiError(Exception):
    def __init__(self, status: int, body: str = ""):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class EmptyResponseError(Exception):
    pass


def api_error_from(err: object) -> ApiError:
    """Wrap an `error` object found inside a response body, keeping the upstream status code."""
    code = err.get("code") if isinstance(err, dict) else None
    return ApiError(code if isinstance(code, int) else 500, json.dumps(err)[:500])


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ApiError):
        return exc.status in RETRY_STATUS or exc.status >= 500
    return isinstance(exc, (httpx.TransportError, EmptyResponseError, json.JSONDecodeError))


# --------------------------------------------------------------------------- request


def format_message(question: dict, model_id: str) -> list[dict]:
    if question["image"]:  # "" if not multi-modal
        content = [
            {"type": "text", "text": question["question"]},
            {"type": "image_url", "image_url": {"url": question["image"]}},
        ]
    else:
        content = question["question"]

    system_role = "user" if O1_RE.search(model_id) else "system"
    return [
        {"role": system_role, "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def reasoning_param(thinking: str) -> dict:
    """OpenRouter unified `reasoning` object. `none` -> disabled; else enabled with an effort."""
    if thinking == "none":
        return {"enabled": False}
    return {"enabled": True, "effort": thinking}  # low | medium | high | max


def build_payload(args: argparse.Namespace, messages: list[dict], session_id: str) -> dict:
    payload: dict = {
        "model": args.model_id,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stream": False,  # the agent API streams anyway; stats come from the final events
        "max_tokens": args.max_completion_tokens,
        "messages": messages,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.api_style == "openrouter":
        payload["reasoning"] = reasoning_param(args.thinking)
        payload["provider"] = {"only": args.provider_list, "allow_fallbacks": False}
        payload["usage"] = {"include": True}  # per-request token counts / cost in the response
    else:  # budget: thinking is simply on or off
        payload.update(web_tools=False, code_tools=False, canvas=False)
        payload["metadata"] = {"user_id": session_id}
        if args.thinking == "on":  # the agent API rejects thinking without an integer budget
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": args.thinking_budget or DEFAULT_THINKING_BUDGET,
            }
    return payload


def resolve_url(base_url: str, endpoint: str | None = None) -> str:
    """(budget style only) Append the endpoint path unless LLM_BASE_URL already contains one."""
    url = base_url.strip().rstrip("/")
    if url.startswith(("http://", "https://")) and ENDPOINT_RE.search(url):
        return url
    path = (endpoint or "").strip("/")
    return f"{url}/{path}" if path else url


@dataclass
class Parsed:
    thinking: str = ""
    text: str = ""
    usage: dict = field(default_factory=dict)  # raw `usage` block (merged over the stream)
    timings: dict = field(default_factory=dict)  # engine `naivert.timings` (agent API)
    finish_reason: str | None = None
    steps: int | None = None


def _content_text(content: object) -> str:
    """Plain text of an OpenAI `content` field (a string or a list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text") or ""
            for p in content
            if isinstance(p, dict) and p.get("type") in (None, "text", "output_text")
        )
    return ""


def _message_texts(messages: list) -> tuple[str, str]:
    """(thinking, text) of the assistant turns in a `messages` array / Anthropic-style message."""
    thinking: list[str] = []
    text: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("thinking", "redacted_thinking"):
                thinking.append(block.get("thinking") or block.get("text") or "")
            elif btype == "text":
                text.append(block.get("text") or "")
    return "".join(thinking), "".join(text)


def _merge_usage(usage: dict, update: object) -> None:
    if isinstance(update, dict):
        usage.update({k: v for k, v in update.items() if v is not None})


def parse_response(data: object) -> Parsed:
    """Parse a non-streamed body: OpenAI/OpenRouter `choices`, Anthropic message or agent `messages`."""
    if not isinstance(data, dict):
        raise EmptyResponseError(f"unexpected response type {type(data).__name__}")
    if err := data.get("error"):
        raise api_error_from(err)

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    naive = data.get("naivert")
    timings = naive["timings"] if isinstance(naive, dict) and isinstance(naive.get("timings"), dict) else {}
    out = Parsed(
        usage={k: v for k, v in usage.items() if v is not None},
        timings=timings,
        steps=data["steps"] if isinstance(data.get("steps"), int) else None,
    )

    thinking: list[str] = []
    text: list[str] = []
    finish: str | None = None
    for choice in data.get("choices") or []:
        if err := choice.get("error"):  # OpenRouter: upstream failure after a 200
            raise api_error_from(err)
        msg = choice.get("message") or {}
        text.append(_content_text(msg.get("content")))
        thinking.append(msg.get("reasoning_content") or msg.get("reasoning") or "")
        finish = choice.get("finish_reason") or finish
    out.text, out.thinking = "".join(text), "".join(thinking)

    if not (out.text or out.thinking):  # Anthropic-style message or agent `messages`
        messages = data["messages"] if isinstance(data.get("messages"), list) else [data]
        out.thinking, out.text = _message_texts(messages)

    out.finish_reason = finish or data.get("stop_reason")
    if out.finish_reason == "error":
        raise ApiError(502, "finish_reason=error")
    return out


def parse_stream(body: str) -> Parsed:
    """Parse an SSE body; the last `message_delta` wins (final usage + engine timings)."""
    out = Parsed()
    blocks: dict[int, list[str]] = {}
    kinds: dict[int, str] = {}  # index -> "thinking" | "text"
    messages: list[dict] = []

    def add(kind: str, index: int, chunk: str | None) -> None:
        if chunk:
            kinds.setdefault(index, kind)
            blocks.setdefault(index, []).append(chunk)

    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if not isinstance(event, dict):
            continue
        if err := event.get("error"):
            raise api_error_from(err)

        message = event.get("message")  # message_start
        if isinstance(message, dict):
            _merge_usage(out.usage, message.get("usage"))

        for choice in event.get("choices") or []:  # OpenAI-style chunks
            delta = choice.get("delta") or {}
            add("text", 0, delta.get("content"))
            add("thinking", 0, delta.get("reasoning_content") or delta.get("reasoning"))
            out.finish_reason = choice.get("finish_reason") or out.finish_reason

        etype = event.get("type")
        delta = event.get("delta") or {}
        index = event.get("index") if isinstance(event.get("index"), int) else 0
        if etype == "content_block_start":
            block = event.get("content_block") or {}
            kinds[index] = "thinking" if block.get("type") in ("thinking", "redacted_thinking") else "text"
        elif etype == "content_block_delta":
            if delta.get("type") == "text_delta":
                add(kinds.get(index, "text"), index, delta.get("text"))
            elif delta.get("type") == "thinking_delta":
                add("thinking", index, delta.get("thinking") or delta.get("text"))
        elif etype == "message_delta":  # the event we want: final usage + engine timings
            out.finish_reason = delta.get("stop_reason") or out.finish_reason
            _merge_usage(out.usage, event.get("usage"))
            naive = event.get("naivert")
            if isinstance(naive, dict) and isinstance(naive.get("timings"), dict):
                out.timings.update(naive["timings"])
        elif etype == "agent_done":
            messages.extend(m for m in (event.get("messages") or []) if isinstance(m, dict))
            if isinstance(event.get("steps"), int):
                out.steps = event["steps"]
            out.finish_reason = out.finish_reason or event.get("stop_reason")

        naive = event.get("naivert")
        if isinstance(naive, dict) and isinstance(naive.get("timings"), dict):
            out.timings.update(naive["timings"])
        _merge_usage(out.usage, event.get("usage"))

    out.text = "".join("".join(blocks[i]) for i in sorted(blocks) if kinds.get(i) == "text")
    out.thinking = "".join("".join(blocks[i]) for i in sorted(blocks) if kinds.get(i) == "thinking")
    if messages:  # the final messages are authoritative when the agent sends them
        final_thinking, final_text = _message_texts(messages)
        out.thinking = final_thinking or out.thinking
        out.text = final_text or out.text
    return out


async def post_completion(client: httpx.AsyncClient, url: str, headers: dict, payload: dict) -> tuple[Parsed, float]:
    """One request. Elapsed = send -> full response body received (parsing excluded)."""
    t0 = time.perf_counter()
    r = await client.post(url, json=payload, headers=headers)
    elapsed = time.perf_counter() - t0
    if r.status_code >= 400:
        raise ApiError(r.status_code, r.text[:500])

    body = r.text
    if body.lstrip().startswith(("data:", "event:")) or "text/event-stream" in r.headers.get("content-type", ""):
        return parse_stream(body), elapsed
    if not body.strip():
        raise EmptyResponseError("empty body")
    return parse_response(json.loads(body)), elapsed


async def request_with_retries(
    client: httpx.AsyncClient, url: str, headers: dict, payload: dict, retries: int, label: str
) -> tuple[Parsed, float, int]:
    """Returns (parsed response, elapsed seconds of the successful attempt, attempts used)."""
    last: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            parsed, elapsed = await post_completion(client, url, headers, payload)
            # Only "empty" if nothing came back AND the API says nothing was generated.
            # (Hidden/truncated reasoning has no text but burned tokens: never retry that.)
            if not (parsed.text or parsed.thinking) and not _num(parsed.usage, "completion_tokens", "output_tokens"):
                raise EmptyResponseError("empty response")
            return parsed, elapsed, attempt
        except Exception as exc:
            last = exc
            if not is_retryable(exc) or attempt > retries:
                break
            delay = min(60.0, 2.0**attempt) * (0.5 + random.random())
            log.warning("%s attempt %d/%d failed (%r); retrying in %.1fs", label, attempt, retries + 1, exc, delay)
            await asyncio.sleep(delay)
    raise RuntimeError(f"giving up: {last!r}") from last


# --------------------------------------------------------------------------- token accounting


def _num(d: object, *keys: str) -> float | None:
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if isinstance(v, int | float) and not isinstance(v, bool):
            return float(v)
    return None


def _int(d: object, *keys: str) -> int | None:
    v = _num(d, *keys)
    return None if v is None else int(v)


def extract_tokens(parsed: Parsed, thinking_requested: bool) -> dict[str, int | None]:
    """Token counts, taken from the API's `usage` block (engine timings as a fallback).

    completion_tokens is all generated tokens (thinking + answer); answer_tokens is the part left
    after thinking. Nothing is estimated from text length.
    """
    usage = parsed.usage
    timings = parsed.timings
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    if not isinstance(details, dict):
        details = {}

    prompt = _int(usage, "prompt_tokens")
    if prompt is None and (inp := _int(usage, "input_tokens")) is not None:
        cached = (_int(usage, "cache_read_input_tokens") or 0) + (_int(usage, "cache_creation_input_tokens") or 0)
        prompt = inp + cached
    completion = _int(usage, "completion_tokens", "output_tokens")
    thinking = _int(details, "reasoning_tokens", "thinking_tokens")
    if thinking is None:
        thinking = _int(usage, "reasoning_tokens", "thinking_tokens")

    # engine timings (agent API) fill whatever the usage block omitted
    segments = timings.get("segments") if isinstance(timings.get("segments"), dict) else {}
    if prompt is None and (prompt_n := _int(timings, "prompt_n")) is not None:
        prompt = prompt_n + (_int(timings, "n_prompt_tokens_cache") or 0)
    if completion is None:
        completion = _int(timings, "predicted_n") or _int(segments.get("total"), "tokens")
    if thinking is None:
        thinking = _int(segments.get("think"), "tokens")

    # thinking off and nothing reasoning-like came back: the whole completion is the answer
    if thinking is None and completion is not None and not thinking_requested and not parsed.thinking:
        thinking = 0

    answer = None
    if completion is not None and thinking is not None and 0 <= thinking <= completion:
        answer = completion - thinking

    total = _int(usage, "total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    return {
        "prompt_tokens": prompt,
        "thinking_tokens": thinking,
        "answer_tokens": answer,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def tokens_per_second(parsed: Parsed, tokens: dict, elapsed: float) -> tuple[float | None, str | None, float | None]:
    """(tokens/s, source, end-to-end tokens/s).

    The engine's own decode speed (`naivert.timings.predicted_per_second`) is preferred when the
    endpoint reports it; otherwise tokens/s is generated tokens over the client-measured time.
    """
    timings = parsed.timings
    server = _num(timings, "predicted_per_second")
    if server is None:
        predicted_n, predicted_ms = _num(timings, "predicted_n"), _num(timings, "predicted_ms")
        if predicted_n is not None and predicted_ms:
            server = predicted_n * 1000.0 / predicted_ms

    generated = tokens.get("completion_tokens")
    e2e = generated / elapsed if generated and elapsed > 0 else None
    e2e = round(e2e, 3) if e2e is not None else None
    if server:
        return round(server, 3), "server", e2e
    if e2e is not None:
        return e2e, "client", e2e
    return None, None, None


# --------------------------------------------------------------------------- jobs


@dataclass
class Result:
    tag: str
    index: int  # index inside the prepared parquet file
    id: str
    ok: bool = False
    elapsed_s: float | None = None  # successful attempt only
    wall_with_retries_s: float | None = None
    tokens_per_s: float | None = None
    tokens_per_s_source: str | None = None  # "server" (engine timings) | "client" (tokens / time)
    e2e_tokens_per_s: float | None = None  # generated tokens / client-measured time
    finish_reason: str | None = None
    attempts: int = 0
    steps: int | None = None
    tokens: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)  # raw usage block from the response
    timings: dict = field(default_factory=dict)  # engine timings subset
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


async def _run_job(
    client: httpx.AsyncClient, args: argparse.Namespace, url: str, api_key: str, res: Result, row: dict
) -> None:
    session_id = f"verbosity-{res.tag}-{row['id']}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Session-Id": session_id,
    }
    payload = build_payload(args, format_message(row, args.model_id), session_id)
    t_start = time.perf_counter()
    parsed, elapsed, attempts = await request_with_retries(
        client, url, headers, payload, args.retries, f"[{res.tag}] #{res.index}"
    )
    res.elapsed_s = round(elapsed, 3)
    res.wall_with_retries_s = round(time.perf_counter() - t_start, 3)
    res.attempts = attempts
    res.finish_reason = parsed.finish_reason
    res.steps = parsed.steps
    res.usage = parsed.usage
    res.timings = {k: parsed.timings[k] for k in TIMING_KEYS if k in parsed.timings}

    res.tokens = extract_tokens(parsed, args.thinking != "none")
    res.tokens_per_s, res.tokens_per_s_source, res.e2e_tokens_per_s = tokens_per_second(parsed, res.tokens, elapsed)

    if not parsed.usage:
        add_warning(res, "no_usage", "response carried no usage block")
    missing = [m for m, v in res.tokens.items() if v is None]
    if missing:
        add_warning(res, "missing_tokens", f"not reported by the API: {', '.join(missing)}")
    if parsed.finish_reason in TRUNCATED_REASONS:
        add_warning(
            res,
            "truncated",
            f"finish_reason={parsed.finish_reason!r}; raise --max-completion-tokens (tokens are capped by the limit)",
        )
    if not parsed.text.strip():
        add_warning(res, "empty_answer", f"no answer text (finish_reason={parsed.finish_reason!r})")
    if attempts > 1:
        add_warning(res, "retried", f"succeeded after {attempts} attempts", logging.INFO)

    if args.verbose:
        log_verbose(res.tag, res.index, parsed.thinking, parsed.text)
    res.ok = True


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
    try:
        await _run_job(client, args, url, api_key, res, row)
    except Exception as exc:  # one bad question must never take the whole run down
        res.ok = False
        res.error = repr(exc)
        log.error("[%s] #%d FAILED permanently: %s", tag, index, res.error)
    return res


def select_indices(n_rows: int, args: argparse.Namespace, tag: str) -> list[int]:
    if args.limit and args.limit > n_rows - (0 if args.subset_seed is not None else args.offset):
        log.warning("[%s] --limit %d exceeds available questions; using fewer", tag, args.limit)
    if args.subset_seed is not None:  # random subset, reproducible and independent per dataset
        k = min(args.limit, n_rows)
        return sorted(random.Random(f"{args.subset_seed}-{tag}").sample(range(n_rows), k))
    end = args.offset + args.limit if args.limit else n_rows
    return list(range(n_rows))[args.offset : end]


async def run_all(args: argparse.Namespace, url: str, api_key: str) -> tuple[list[Result], float]:
    """Returns (results, total wall-clock seconds of the whole run)."""
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

    # The read timeout must cover the WHOLE response (the body is read as one piece).
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        await asyncio.gather(*(worker(client) for _ in range(args.workers)))
    wall = time.perf_counter() - t0
    if not args.verbose:
        print(file=sys.stderr)
    order = list(DATASETS)
    return sorted(results, key=lambda r: (order.index(r.tag), r.index)), wall


# --------------------------------------------------------------------------- stats


def metric_value(r: Result, metric: str) -> float | None:
    if metric == "elapsed_s":
        return r.elapsed_s
    if metric == "tokens_per_s":
        return r.tokens_per_s
    if metric == "e2e_tokens_per_s":
        return r.e2e_tokens_per_s
    return r.tokens.get(metric)


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


def summarize_group(rs: list[Result], level: str) -> dict:
    ok = [r for r in rs if r.ok]
    metrics: dict[str, dict | None] = {}
    for m in STAT_METRICS:
        values = [v for r in ok if (v := metric_value(r, m)) is not None]
        metrics[m] = summarize(values) if values else None

    # aggregate throughput = sum(tokens) / sum(time): long requests weigh more than in the mean of ratios
    timed = [r for r in ok if r.tokens.get("completion_tokens") and r.elapsed_s]
    time_sum = sum(r.elapsed_s for r in timed)
    aggregate = sum(r.tokens["completion_tokens"] for r in timed) / time_sum if timed and time_sum else None
    return {
        "difficulty": level,
        "n": len(rs),
        "ok": len(ok),
        "failed": len(rs) - len(ok),
        "indices": [r.index for r in rs],
        "tokens_per_s_sources": dict(Counter(r.tokens_per_s_source for r in ok if r.tokens_per_s_source)),
        "warning_counts": dict(Counter(c for r in rs for c in r.warnings)),
        "aggregate_tokens_per_s": aggregate,
        "metrics": metrics,
    }


def build_summary(results: list[Result], tags: list[str]) -> dict:
    summary = {tag: summarize_group([r for r in results if r.tag == tag], DATASETS[tag][1]) for tag in tags}
    if len(tags) > 1:
        summary["ALL"] = summarize_group(results, "-")
    return summary


def _cell(stat: dict | None, digits: int) -> str:
    return "n/a" if not stat else f"{stat['mean']:.{digits}f} ({stat['median']:.{digits}f})"


def print_table(summary: dict, run: dict, args: argparse.Namespace) -> None:
    route = ",".join(args.provider_list) if args.api_style == "openrouter" else "budget API"
    head = f"{args.model_id} | {route} | top_p={args.top_p} temp={args.temperature} thinking={args.thinking}"

    speed = Table(
        title=f"{head}\nTime and speed",
        caption=(
            "cells: mean (median) | Tok/s = engine decode speed when reported, else generated tokens / "
            "request time | Tok/s agg = sum(tokens) / sum(time)"
        ),
    )
    for name, just in [
        ("Dataset", "left"),
        ("Level", "left"),
        ("OK/N", "right"),
        ("Time/req (s)", "right"),
        ("Tok/s", "right"),
        ("Tok/s e2e", "right"),
        ("Tok/s agg", "right"),
        ("Completion tok", "right"),
    ]:
        speed.add_column(name, justify=just)

    tokens = Table(
        title="Tokens per request",
        caption="cells: mean (median) | Completion = Thinking + Answer (all generated tokens)",
    )
    for name, just in [
        ("Dataset", "left"),
        ("Prompt tok", "right"),
        ("Thinking tok", "right"),
        ("Answer tok (post-thinking)", "right"),
        ("Completion tok", "right"),
        ("Total tok", "right"),
    ]:
        tokens.add_column(name, justify=just)

    for tag, s in summary.items():
        m = s["metrics"]
        agg = s["aggregate_tokens_per_s"]
        speed.add_row(
            tag,
            s["difficulty"],
            f"{s['ok']}/{s['n']}",
            _cell(m["elapsed_s"], 2),
            _cell(m["tokens_per_s"], 1),
            _cell(m["e2e_tokens_per_s"], 1),
            f"{agg:.1f}" if agg is not None else "n/a",
            _cell(m["completion_tokens"], 0),
        )
        tokens.add_row(
            tag,
            _cell(m["prompt_tokens"], 0),
            _cell(m["thinking_tokens"], 0),
            _cell(m["answer_tokens"], 0),
            _cell(m["completion_tokens"], 0),
            _cell(m["total_tokens"], 0),
        )
    console.print(speed)
    console.print(tokens)
    console.print(
        f"Run wall-clock: {run['total_wall_s']:.1f}s for {run['n_requests']} requests, {run['workers']} worker(s)"
    )


def print_warnings(results: list[Result]) -> None:
    failures = [r for r in results if not r.ok]
    counts = Counter(c for r in results for c in r.warnings)
    if not counts and not failures:
        console.print("\n[green]No warnings.[/green]")
        return

    if counts:
        table = Table(title="Warnings")
        table.add_column("Warning", style="yellow")
        table.add_column("Count", justify="right")
        table.add_column("Meaning")
        for code, n in counts.most_common():
            table.add_row(code, str(n), WARNING_DESCRIPTIONS.get(code, ""))
        console.print(table)
    if failures:
        console.print(f"[red]{len(failures)} request(s) failed permanently[/red]")
        for r in failures[:10]:
            console.print(f"  - [{r.tag}] #{r.index}: {r.error}")
        if len(failures) > 10:
            console.print(f"  ... and {len(failures) - 10} more")


# --------------------------------------------------------------------------- cli


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)


def output_path(args: argparse.Namespace) -> Path:
    route = _safe("+".join(args.provider_list)) if args.api_style == "openrouter" else "budget"
    name = f"{_safe(args.model_id)}_{route}_{args.top_p}_{args.temperature}_thinking_{args.thinking}.json"
    return args.output_dir / name


def thinking_type(value: str) -> str:
    v = value.lower()
    v = THINKING_ALIASES.get(v, v)
    if v not in (*THINKING_CHOICES, "on"):
        raise argparse.ArgumentTypeError(
            f"choose from none|low|mid|high|max (openrouter) or on|off (budget); got {value!r}"
        )
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
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument(
        "--thinking",
        type=thinking_type,
        default="none",
        help="openrouter: none|low|mid|high|max ; budget: on|off (none = off)",
    )
    p.add_argument(
        "--api-style",
        choices=(*API_STYLES, "openai"),  # 'openai' = deprecated alias of 'openrouter'
        default="openrouter",
    )
    p.add_argument(
        "--provider",
        default=None,
        help=f"openrouter only: provider slug(s), comma-separated (default: {DEFAULT_PROVIDER}); "
        "fallbacks to other providers are always disabled",
    )
    p.add_argument(
        "--endpoint", default=None, help=f"budget only: path appended to LLM_BASE_URL (default: {BUDGET_ENDPOINT})"
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help=f"budget only: budget_tokens sent when thinking is on (default: {DEFAULT_THINKING_BUDGET})",
    )
    p.add_argument("--max-completion-tokens", type=int, default=DEFAULT_MAX_COMPLETION_TOKENS)
    p.add_argument("--seed", type=int, default=None, help="sampling seed, sent only if set")
    p.add_argument("--workers", "-n", type=int, default=1)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--timeout", type=float, default=3600.0, help="per-request timeout in seconds")

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

    if args.api_style == "openai":
        args.api_style = "openrouter"
    if args.limit < 0 or args.offset < 0:
        p.error("--limit and --offset must be >= 0")
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.thinking_budget is not None and args.thinking_budget < 1:
        p.error("--thinking-budget must be >= 1")
    if args.subset_seed is not None and not args.limit:
        p.error("--subset-seed needs --limit")
    if args.subset_seed is not None and args.offset:
        p.error("--subset-seed and --offset are mutually exclusive")

    if args.api_style == "openrouter":
        if args.thinking == "on":
            p.error("--thinking on is for --api-style budget; use none|low|mid|high|max with openrouter")
        for flag in ("endpoint", "thinking_budget"):
            if getattr(args, flag) is not None:
                p.error(f"--{flag.replace('_', '-')} only applies to --api-style budget")
        args.provider = args.provider or DEFAULT_PROVIDER
        args.provider_list = [x.strip() for x in args.provider.split(",") if x.strip()]
        if not args.provider_list:
            p.error("--provider is empty")
    else:
        if args.thinking not in ("none", "on"):
            p.error("--api-style budget only supports --thinking on|off")
        if args.provider is not None:
            p.error("--provider only applies to --api-style openrouter")
        args.provider_list = []

    args.datasets = list(dict.fromkeys(args.datasets))  # dedupe, keep order
    return args


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        sys.exit("LLM_API_KEY must be set in the environment")
    if args.api_style == "openrouter":
        url = OPENROUTER_URL
    else:
        base_url = os.environ.get("LLM_BASE_URL")
        if not base_url:
            sys.exit("LLM_BASE_URL must be set for --api-style budget")
        url = resolve_url(base_url, args.endpoint or BUDGET_ENDPOINT)

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.workers > 1:
        log.warning(
            "workers=%d: concurrent requests compete for the endpoint, so per-request tokens/s will be "
            "lower than the single-stream speed (use --workers 1 for clean speed numbers)",
            args.workers,
        )

    log.info("using endpoint %s (api-style=%s)", url, args.api_style)
    results, wall = asyncio.run(run_all(args, url, api_key))
    run = {"total_wall_s": round(wall, 3), "n_requests": len(results), "workers": args.workers}
    summary = build_summary(results, args.datasets)
    print_table(summary, run, args)
    print_warnings(results)

    path = output_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(url=url, timestamp=datetime.now(UTC).isoformat())
    path.write_text(
        json.dumps(
            {"config": config, "run": run, "summary": summary, "results": [asdict(r) for r in results]},
            indent=2,
        )
    )
    print(f"saved stats -> {path}")


if __name__ == "__main__":
    main()
