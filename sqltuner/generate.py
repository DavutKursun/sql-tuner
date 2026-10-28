"""Backends that turn chat messages into replies: a local Hugging Face model (optionally with a LoRA adapter), the
Anthropic API (reference baselines, with cost tracking and a spend cap), llama.cpp GGUF (serving, M5) and `gold`
(an oracle used only to test the harness)."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from .config import API_MODELS, API_THINKING_OFF, GEN, GenRules


@dataclass
class Reply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    ms: list[float] = field(default_factory=list)

    def add(self, r: Reply) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cost_usd += r.cost_usd
        self.ms.append(r.ms)

    def as_dict(self) -> dict:
        import numpy as np

        ms = np.asarray(self.ms) if self.ms else np.zeros(1)
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, "cost_usd": round(self.cost_usd, 4),
                "cost_per_1000_questions_usd": round(1000 * self.cost_usd / max(self.calls, 1), 4), "latency_ms_p50": float(np.percentile(ms, 50)),
                "latency_ms_p95": float(np.percentile(ms, 95))}


class Backend:
    name = "base"

    def generate(self, batch: list[list[dict]], n: int = 1, temperature: float = 0.0, max_new_tokens: int = GEN.max_new_tokens) -> list[list[Reply]]:
        """For each conversation, `n` replies."""
        raise NotImplementedError


# --- local Hugging Face model -----------------------------------------------------------------------------------------
def pick_device(name: str | None = None) -> str:
    import torch

    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cuda_dtype():
    """bf16 on GPUs that have it natively (A100, L4, H100: compute capability 8+), else fp16 (T4). torch's own
    `is_bf16_supported()` also counts emulated bf16 by default, which says True on a T4 and would be very slow there."""
    import torch

    try:
        native = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # torch without the argument
        native = torch.cuda.get_device_capability()[0] >= 8
    return torch.bfloat16 if native else torch.float16


def amp_dtype(device: str):
    """Autocast dtype for training on `device` (None off CUDA). A GradScaler is needed exactly when this is fp16."""
    return cuda_dtype() if device == "cuda" else None


def load_model(model_id: str, adapter: str | None = None, device: str | None = None, dtype: str = "auto", trainable: bool = False):
    """Tokenizer + causal LM (+ LoRA adapter). On CUDA bf16 where native, else fp16 (`cuda_dtype`); fp16 for inference
    on MPS; fp32 on CPU and for training off CUDA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    if dtype == "auto":
        torch_dtype = cuda_dtype() if device == "cuda" else (torch.float16 if device == "mps" and not trainable else torch.float32)
    else:
        torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # generation pads on the left
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    model.to(device)
    if not trainable:
        model.eval()
    return tok, model, device


def sampling_kwargs(temperature: float) -> dict:
    """Plain argmax (temperature 0) or plain sampling at `temperature`, whatever the model's generation_config says:
    Qwen2.5-Coder ships repetition_penalty 1.05–1.1, which generate() would otherwise apply even when greedy. GRPO
    depends on this too: its samples must come from the distribution whose log-probs it trains on."""
    if temperature > 0:
        return {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
    return {"do_sample": False, "temperature": None, "top_p": None, "top_k": None, "repetition_penalty": 1.0}


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_id: str, adapter: str | None = None, device: str | None = None, batch_size: int = 8, tok=None, model=None) -> None:
        if model is None:
            tok, model, device = load_model(model_id, adapter, device)
        self.tok, self.model, self.device = tok, model, device or pick_device()
        self.batch_size = batch_size
        self.model_id, self.adapter = model_id, adapter

    def generate(self, batch, n=1, temperature=0.0, max_new_tokens=GEN.max_new_tokens):
        import torch

        from .prompts import apply_chat_template

        out: list[list[Reply]] = []
        self.model.eval()
        self.tok.padding_side = "left"
        for start in range(0, len(batch), self.batch_size):
            convs = batch[start : start + self.batch_size]
            texts = [apply_chat_template(self.tok, c) for c in convs]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=max_new_tokens,
                                          **sampling_kwargs(temperature), num_return_sequences=n, pad_token_id=self.tok.pad_token_id)
            ms = (time.perf_counter() - t0) * 1000 / (len(convs) * n)
            new = gen[:, enc["input_ids"].shape[1] :]
            prompt_lens = enc["attention_mask"].sum(1).tolist()
            for i in range(len(convs)):
                group = []
                for j in range(n):
                    ids = new[i * n + j]
                    comp = int((ids != self.tok.pad_token_id).sum())
                    group.append(Reply(self.tok.decode(ids, skip_special_tokens=True), int(prompt_lens[i]), comp, ms))
                out.append(group)
        return out


# --- Anthropic API ----------------------------------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    pass


class AnthropicBackend(Backend):
    """Reference baselines. The key is read from the environment (.env), never printed. The system prompt and few-shot
    turns are marked for prompt caching. `max_usd` stops the run before it can overspend. No sampling parameters are
    sent (the current models reject them), so `temperature` is ignored; thinking is off where config says so."""

    name = "anthropic"

    def __init__(self, model_id: str, max_usd: float = 5.0, client=None) -> None:
        if client is None:
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in .env; never paste it into a chat)")
            client = anthropic.Anthropic()
        self.client, self.model_id, self.max_usd = client, model_id, max_usd
        self.prices = API_MODELS.get(model_id, (0.0, 0.0))
        self.thinking = API_THINKING_OFF.get(model_id)
        self.spent = 0.0

    def _one(self, conv: list[dict], temperature: float, max_new_tokens: int) -> Reply:
        if self.spent >= self.max_usd:
            raise BudgetExceeded(f"spend cap {self.max_usd} USD reached")
        system = [{"type": "text", "text": conv[0]["content"], "cache_control": {"type": "ephemeral"}}]
        msgs = [dict(m) for m in conv[1:]]
        if len(msgs) > 1:  # few-shot turns are identical for every question: cache them too
            msgs[-2] = {"role": msgs[-2]["role"], "content": [{"type": "text", "text": msgs[-2]["content"], "cache_control": {"type": "ephemeral"}}]}
        t0 = time.perf_counter()
        extra = {"thinking": self.thinking} if self.thinking else {}
        r = self.client.messages.create(model=self.model_id, system=system, messages=msgs, max_tokens=max_new_tokens, **extra)
        ms = (time.perf_counter() - t0) * 1000
        u = r.usage
        cached = (getattr(u, "cache_read_input_tokens", 0) or 0)
        written = (getattr(u, "cache_creation_input_tokens", 0) or 0)
        p_in, p_out = self.prices
        cost = (u.input_tokens * p_in + written * p_in * 1.25 + cached * p_in * 0.1 + u.output_tokens * p_out) / 1e6
        self.spent += cost
        text = "".join(b.text for b in r.content if getattr(b, "type", "text") == "text")  # by block type: a reply may start with thinking
        return Reply(text, u.input_tokens + cached + written, u.output_tokens, ms, cost)

    def generate(self, batch, n=1, temperature=0.0, max_new_tokens=GEN.max_new_tokens):
        return [[self._one(c, temperature, max_new_tokens) for _ in range(n)] for c in batch]


# --- oracle (tests only) ------------------------------------------------------------------------------------------------
class GoldBackend(Backend):
    """Answers with the gold SQL it is handed; checks that the harness scores a perfect model as perfect."""

    name = "gold"

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.i = 0

    def generate(self, batch, n=1, temperature=0.0, max_new_tokens=GEN.max_new_tokens):
        out = []
        for _ in batch:
            out.append([Reply(f"```sql\n{self.answers[self.i]}\n```") for _ in range(n)])
            self.i += 1
        return out


# --- llama.cpp (serving) ----------------------------------------------------------------------------------------------
class LlamaCppBackend(Backend):
    """GGUF model on CPU via llama-cpp-python (installed only for serving)."""

    name = "llamacpp"

    def __init__(self, gguf_path: str, n_ctx: int = 4096, n_threads: int | None = None) -> None:
        from llama_cpp import Llama

        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)

    def generate(self, batch, n=1, temperature=0.0, max_new_tokens=GEN.max_new_tokens):
        out = []
        for conv in batch:
            group = []
            for _ in range(n):
                t0 = time.perf_counter()
                r = self.llm.create_chat_completion(messages=conv, temperature=temperature, max_tokens=max_new_tokens)
                u = r.get("usage", {})
                group.append(Reply(r["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0), (time.perf_counter() - t0) * 1000))
            out.append(group)
        return out


def make_backend(spec: str, adapter: str | None = None, device: str | None = None, max_usd: float = 5.0, batch_size: int = 8) -> Backend:
    """spec: 'hf:<model id or path>' | 'anthropic:<model id>' | 'llamacpp:<path.gguf>'."""
    kind, _, ident = spec.partition(":")
    if kind == "hf":
        return HFBackend(ident, adapter, device, batch_size)
    if kind == "anthropic":
        return AnthropicBackend(ident, max_usd)
    if kind == "llamacpp":
        return LlamaCppBackend(ident)
    raise ValueError(f"unknown backend spec {spec!r}")


GEN_DEFAULTS: GenRules = GEN
