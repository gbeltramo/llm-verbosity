import dataclasses
from pathlib import Path

from .constants import VALID_THINKING_EFFORTS


@dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
class DatasetConfig:
    datasets: list[str]
    len_subset: int
    offset: int
    random_seed: int
    data_dir: Path


@dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
class OpenRouterConfig:
    api_key: str
    model_id: str
    temperature: float
    top_p: float
    thinking_effort: VALID_THINKING_EFFORTS
    inference_provider: str
    random_seed: int
    verbose: bool = False


@dataclasses.dataclass
class PendingGeneration:
    dataset: str
    index: int
    completion: dict
    client_elapsed: float


@dataclasses.dataclass(slots=True)
class BenchmarkResult:
    dataset: str
    index: int
    answer: str
    reasoning: str
    provider: str
    generation_time_ms: float | None
    server_latency_ms: float | None
    native_prompt_tokens: int | None
    native_completion_tokens: int | None
    native_reasoning_tokens: int | None
    or_prompt_tokens: int | None
    or_completion_tokens: int | None
    cost: float
    client_elapsed_seconds: float
    model_id: str = ""
    thinking_effort: str = ""

    # ---- derived: tokens ----
    @property
    def native_answer_tokens(self) -> int | None:
        if self.native_completion_tokens is None:
            return None
        return max(self.native_completion_tokens - (self.native_reasoning_tokens or 0), 0)

    # ---- derived: chars ----
    @property
    def reasoning_chars(self) -> int:
        return len(self.reasoning)

    @property
    def answer_chars(self) -> int:
        return len(self.answer)

    @property
    def completion_chars(self) -> int:
        """Thinking + answer."""
        return self.reasoning_chars + self.answer_chars

    # ---- derived: throughput ----
    def _per_second(self, amount: int | float | None) -> float | None:
        if not self.generation_time_ms or not amount:
            return None
        return amount / (self.generation_time_ms / 1000)

    @property
    def native_tps(self) -> float | None:
        return self._per_second(self.native_completion_tokens)

    @property
    def or_tps(self) -> float | None:
        return self._per_second(self.or_completion_tokens)

    @property
    def chars_per_second(self) -> float | None:
        return self._per_second(self.completion_chars)

    def to_row(self, include_text: bool = True) -> dict:
        row = {
            "model_id": self.model_id,
            "thinking_effort": self.thinking_effort,
            "dataset": self.dataset,
            "index": self.index,
            "provider": self.provider,
            "generation_time_ms": self.generation_time_ms,
            "server_latency_ms": self.server_latency_ms,
            "native_prompt_tokens": self.native_prompt_tokens,
            "native_completion_tokens": self.native_completion_tokens,
            "native_reasoning_tokens": self.native_reasoning_tokens,
            "native_answer_tokens": self.native_answer_tokens,
            "or_prompt_tokens": self.or_prompt_tokens,
            "or_completion_tokens": self.or_completion_tokens,
            "reasoning_chars": self.reasoning_chars,
            "answer_chars": self.answer_chars,
            "completion_chars": self.completion_chars,
            "native_tps": self.native_tps,
            "or_tps": self.or_tps,
            "chars_per_second": self.chars_per_second,
            "cost": self.cost,
            "client_elapsed_seconds": self.client_elapsed_seconds,
        }
        if include_text:
            row["reasoning"] = self.reasoning
            row["answer"] = self.answer
        return row
