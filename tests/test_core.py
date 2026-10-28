"""Splits, metrics, the evaluation runner, the API backend (mocked), SFT label masking and learning, and the GRPO maths."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from sqltuner.config import GRPO, LORA, PROMPT, SFT, ExecRules, SplitRules
from sqltuner.data import CLOSED_SPLITS, build_splits, load_split, save_splits
from sqltuner.db import ExecResult
from sqltuner.generate import AnthropicBackend, BudgetExceeded, GoldBackend, HFBackend, Reply, load_model
from sqltuner.metrics import accuracy, majority_vote, paired, reward
from sqltuner.prompts import apply_chat_template, messages
from sqltuner.runner import evaluate, few_shot_pool, summarize


# --- splits -----------------------------------------------------------------------------------------------------------
def test_splits_are_database_disjoint(spider):
    s = spider["summary"]
    assert s["db_overlap_train_vs_val_dev"] == [] and len(s["val_dbs"]) >= 1
    assert s["splits"]["train"]["questions"] > 0 and s["splits"]["dev"]["questions"] == 28 and s["splits"]["test"]["questions"] == 14
    train = load_split("train", spider["splits_dir"])
    val = load_split("val", spider["splits_dir"])
    assert not ({e.db_id for e in train} & {e.db_id for e in val})
    assert all(e.split == "val" for e in val) and all(e.hardness in ("easy", "medium", "hard", "extra") for e in train)
    assert set(CLOSED_SPLITS) == {"dev", "test"}
    with pytest.raises(FileNotFoundError):
        load_split("val", spider["splits_dir"] / "nope")


def test_empty_databases_and_truncated_gold_are_flagged(spider, tmp_path):
    s = spider["summary"]
    assert s["empty_dbs"] == ["geo_1"] and s["splits"]["train"]["db_empty"] == 14 and s["splits"]["val"]["db_empty"] == 0
    train = load_split("train", spider["splits_dir"])
    assert {e.db_id for e in train if e.db_empty} == {"geo_1"} and any(e.gold_empty for e in train if e.db_empty)
    assert s["splits"]["train"]["gold_empty_on_nonempty_dbs"] == sum(e.gold_empty and not e.db_empty for e in train)
    assert s["splits"]["train"]["gold_truncated"] == 0
    small = save_splits(build_splits(spider["sd"], SplitRules(val_min_questions=20, val_max_dbs=3), log=None, exec_rules=ExecRules(max_rows=5)), tmp_path)
    assert small["splits"]["train"]["gold_truncated"] > 0  # "List the names of all …" returns 30 rows


# --- metrics ----------------------------------------------------------------------------------------------------------
def test_accuracy_and_paired_intervals():
    rng = np.random.default_rng(0)
    dbs = np.repeat(np.arange(20), 25)
    a = rng.random(500) < 0.6
    b = rng.random(500) < 0.4
    acc = accuracy(a, dbs, n_resamples=300)
    assert acc["ci_case"][0] < acc["ex"] < acc["ci_case"][1] and acc["n_db"] == 20
    assert (acc["ci_db"][1] - acc["ci_db"][0]) > 0  # database-level interval exists
    p = paired(a, b, dbs, n_resamples=300)
    assert p["diff"] == pytest.approx(a.mean() - b.mean()) and p["significant_case"]
    same = paired(a, a, dbs, n_resamples=50)
    assert same["diff"] == 0 and not same["significant_case"]


def test_majority_vote_and_reward():
    good = ExecResult(True, [(1,)], ["c"])
    other = ExecResult(True, [(2,)], ["c"])
    bad = ExecResult(False, error="sqlite: x")
    assert majority_vote([bad, other, good, good]) == 2 and majority_vote([bad, bad]) == 0
    gold = ExecResult(True, [(1,), (2,)], ["c"])
    assert reward("", bad, gold, "SELECT c FROM t").value == 0.0
    r = reward("SELECT c FROM t", ExecResult(True, [(2,), (1,)], ["c"]), gold, "SELECT c FROM t")
    assert r.value == 1.0 and r.correct
    r = reward("SELECT c FROM t LIMIT 1", ExecResult(True, [], ["c"]), gold, "SELECT c FROM t", exec_bonus=0.1)
    assert r.value == 0.1 and not r.correct and "empty_result" in r.flags and "added_limit" in r.flags
    r = reward("DROP TABLE t", ExecResult(False, error="guard: not a query"), gold, "SELECT c FROM t")
    assert r.value == 0.0 and "guard_refused" in r.flags


# --- runner -----------------------------------------------------------------------------------------------------------
def test_runner_scores_gold_as_perfect_and_constant_as_wrong(spider):
    val = load_split("val", spider["splits_dir"])
    preds, usage = evaluate(val, GoldBackend([e.query for e in val]), spider["sd"], log=None)
    assert all(p.correct for p in preds) and usage.calls == len(val)
    s = summarize(preds, usage)
    assert s["accuracy"]["ex"] == 1.0 and s["executed_rate"] == 1.0
    preds, _ = evaluate(val, GoldBackend(["SELECT 12345"] * len(val)), spider["sd"], log=None)
    assert np.mean([p.correct for p in preds]) == 0.0 and all(p.executed for p in preds)
    shots = few_shot_pool(load_split("train", spider["splits_dir"]), spider["sd"], 2)
    assert len(shots) == 2 and len({s[0] for s in shots}) == 2


def test_error_types():
    from sqltuner.runner import error_type

    assert error_type("sqlite: no such column: T3.name") == "unknown column" and error_type("sqlite: no such table: foo") == "unknown table"
    assert error_type('sqlite: near "FROM": syntax error') == "syntax error" and error_type("sqlite: incomplete input") == "incomplete query"
    assert error_type("guard: more than one statement") == "guard: more than one statement" and error_type("timeout") == "timeout"
    assert error_type("sqlite: wrong number of arguments to function substr()") == "sqlite: wrong number of"


def test_ex_on_nonempty_gold(spider):
    on_empty_db = [e for e in load_split("train", spider["splits_dir"]) if e.db_empty]
    preds, _ = evaluate(on_empty_db, GoldBackend(["SELECT 'x' WHERE 0"] * len(on_empty_db)), spider["sd"], log=None)
    assert [p.gold_empty for p in preds] == [e.gold_empty for e in on_empty_db]
    s = summarize(preds)
    # a query that returns nothing matches every one-column empty gold result (Spider's convention), never a non-empty one
    assert s["accuracy"]["ex"] > 0 and s["ex_nonempty_gold"]["ex"] == 0
    assert s["ex_nonempty_gold"]["n"] == sum(not e.gold_empty for e in on_empty_db) > 0 and {"ci_case", "ci_db"} <= set(s["ex_nonempty_gold"])
    assert summarize([p for p in preds if p.gold_empty])["ex_nonempty_gold"] is None
    assert not any(db == "geo_1" for db, *_ in few_shot_pool(load_split("train", spider["splits_dir"]), spider["sd"], 9))


def test_anthropic_backend_costs_cache_and_cap():
    calls = []

    class FakeMessages:
        def create(self, **kw):
            calls.append(kw)
            usage = SimpleNamespace(input_tokens=1000, output_tokens=50, cache_read_input_tokens=2000, cache_creation_input_tokens=0)
            return SimpleNamespace(content=[SimpleNamespace(text="```sql\nSELECT 1\n```")], usage=usage)

    be = AnthropicBackend("claude-haiku-5-5", max_usd=0.0003, client=SimpleNamespace(messages=FakeMessages()))
    conv = messages("CREATE TABLE t (a INT);", "q", [("CREATE TABLE s (b INT);", "q0", "SELECT b FROM s")])
    r = be.generate([conv])[0][0]
    assert r.text.startswith("```sql") and r.cost_usd == pytest.approx((1000 * 0.10 + 2000 * 0.10 * 0.1 + 50 * 0.50) / 1e6)
    assert calls[0]["system"][0]["cache_control"]["type"] == "ephemeral"
    assert isinstance(calls[0]["messages"][-2]["content"], list)  # the few-shot turn before the question is cached
    assert "temperature" not in calls[0] and calls[0]["thinking"] == {"type": "disabled"}  # 5.5 models: no sampling parameters
    assert AnthropicBackend("claude-sonnet-5-5", client=be.client).thinking == {"type": "between_tools"}
    with pytest.raises(BudgetExceeded):
        for _ in range(10):
            be.generate([conv])


def test_spend_cap_keeps_the_questions_already_paid_for(spider):
    class FakeMessages:
        def create(self, **kw):
            usage = SimpleNamespace(input_tokens=1000, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="```sql\nSELECT 1\n```")], usage=usage)

    val = load_split("val", spider["splits_dir"])
    be = AnthropicBackend("claude-haiku-5-5", max_usd=0.002, client=SimpleNamespace(messages=FakeMessages()))  # $0.000125 a call
    preds, usage = evaluate(val, be, spider["sd"], chunk=4, log=None)
    assert 0 < len(preds) < len(val) and len(preds) % 4 == 0 and usage.cost_usd >= 0.002


# --- SFT --------------------------------------------------------------------------------------------------------------
def test_sft_label_masking_and_learning(spider, tiny, tmp_path):

    from sqltuner.sft import IGNORE, add_lora, build_examples, encode, train_sft

    tok, model, dev = load_model(str(tiny), device="cpu", trainable=True)
    msgs = messages("CREATE TABLE t (a INT);", "How many?")
    enc = encode(tok, msgs, "```sql\nSELECT count(*) FROM t\n```")
    n_prompt = len(tok(apply_chat_template(tok, msgs), add_special_tokens=False)["input_ids"])
    assert all(x == IGNORE for x in enc.labels[:n_prompt]) and all(x != IGNORE for x in enc.labels[n_prompt:])
    assert tok.decode([x for x in enc.labels if x != IGNORE]) == "```sql\nSELECT count(*) FROM t\n```" + tok.eos_token
    train_ex = load_split("train", spider["splits_dir"])[:12]
    data, stats = build_examples(train_ex, tok, spider["sd"], PROMPT, max_len=4096)
    assert stats["kept"] == 12 and stats["dropped_too_long"] == 0
    model = add_lora(model, replace(LORA, r=8, alpha=16))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < trainable < sum(p.numel() for p in model.parameters())
    info = train_sft(model, tok, data, data[:4], tmp_path / "sft", "cpu", replace(SFT, epochs=30, lr=1e-2, batch_size=4, grad_accum=1, eval_every=10),
                     gradient_checkpointing=False, log=None, max_steps=60)
    losses = [r["val_loss"] for r in info["log"]]
    assert losses[-1] < 0.7 * losses[0], losses
    assert (tmp_path / "sft" / "adapter_config.json").exists()
    be = HFBackend(str(tiny), adapter=str(tmp_path / "sft"), device="cpu", batch_size=4)
    out = be.generate([msgs, msgs], n=2, temperature=0.7, max_new_tokens=8)
    assert len(out) == 2 and len(out[0]) == 2 and isinstance(out[0][0], Reply)


def test_precision_rule(monkeypatch):
    import torch

    from sqltuner import generate

    # a T4: no native bf16, but torch's default check (emulation included) would say yes
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: including_emulation)
    assert generate.cuda_dtype() == torch.float16 and generate.amp_dtype("cuda") == torch.float16
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)  # A100 / L4 / H100
    assert generate.cuda_dtype() == torch.bfloat16 and generate.amp_dtype("cpu") is None
    import inspect

    from sqltuner.hub import push_folder

    assert inspect.signature(push_folder).parameters["private"].default is True  # adapters go to private repos


def test_sft_loss_on_target_tokens_only_equals_the_full_loss(tiny):
    import torch

    from sqltuner.sft import collate, encode, loss_on

    tok, model, _ = load_model(str(tiny), device="cpu")
    a = encode(tok, messages("CREATE TABLE t (a INT);", "How many?"), "```sql\nSELECT count(*) FROM t\n```")
    b = encode(tok, messages("CREATE TABLE singer (id INT, name TEXT, age INT);", "Names of singers older than 30?"), "```sql\nSELECT name FROM singer WHERE age > 30\n```")
    ids, mask, labels = collate([a, b], tok.pad_token_id)  # different lengths: right padding
    with torch.no_grad():
        total, n = loss_on(model, (ids, mask, labels), "cpu")
        hf = model(input_ids=ids, attention_mask=mask, labels=labels).loss  # HF's own shift-and-mask, mean over targets
    assert n == sum(x != -100 for x in a.labels) + sum(x != -100 for x in b.labels) and torch.allclose(total / n, hf, atol=1e-5)


def test_length_budget_choice():
    from sqltuner.sft import choose_budget

    lengths = [1000] * 90 + [2500] * 8 + [3500] * 2  # 2,048 keeps 90%, 3,072 keeps 98%, 4,096 keeps 100%
    b, r = choose_budget(lengths, lambda b: None)
    assert b == 3072 and r["kept_share"] == {2048: 0.9, 3072: 0.98, 4096: 1.0}
    b, r = choose_budget(lengths, lambda b: {3072: 14.5, 2048: 9.0}[b])  # the 98% budget does not fit: the largest that does
    assert b == 2048 and r["probe_gib"] == {3072: 14.5, 2048: 9.0}
    b, _ = choose_budget([5000] * 10 + [100] * 90, lambda b: 3.0)  # nothing reaches 98%: the largest budget, if it fits
    assert b == 4096


def test_memory_probe_leaves_the_weights_alone(tiny):
    import torch

    from sqltuner.sft import add_lora, encode, memory_probe

    tok, model, _ = load_model(str(tiny), device="cpu", trainable=True)
    model = add_lora(model, replace(LORA, r=4, alpha=8))
    data = [encode(tok, messages("CREATE TABLE t (a INT);", f"Question {i}?" + " pad" * i), "```sql\nSELECT a FROM t\n```") for i in range(6)]
    before = {k: v.clone() for k, v in model.state_dict().items() if "lora" in k}
    r = memory_probe(model, tok, data, "cpu", steps=3, batch_size=2, gradient_checkpointing=False)
    assert r["steps"] == 3 and r["tokens"] == sorted(r["tokens"], reverse=True) and r["peak_reserved_gib"] is None
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items() if "lora" in k)


def test_sft_eta_switches_to_one_epoch(spider, tiny, tmp_path):
    from sqltuner.sft import add_lora, build_examples, train_sft

    tok, model, _ = load_model(str(tiny), device="cpu", trainable=True)
    model = add_lora(model, replace(LORA, r=4, alpha=8))
    data, _ = build_examples(load_split("train", spider["splits_dir"])[:12], tok, spider["sd"], PROMPT, max_len=4096)
    info = train_sft(model, tok, data, data[:2], tmp_path / "sft", "cpu", replace(SFT, epochs=2, batch_size=4, grad_accum=1, eval_every=100),
                     gradient_checkpointing=False, log=None, max_hours=1e-9, eta_after=1, log_every=1)
    assert info["eta"]["switched_to_one_epoch"] and info["epochs_run"] == 1 and info["steps"] == 3  # 12 examples / 4 per step
    assert [r["step"] for r in info["train_log"]] == [1, 2, 3] and info["log"][0]["step"] == 0 and info["precision"] == "fp32"


# --- GRPO -------------------------------------------------------------------------------------------------------------
def test_advantages_and_loss():
    import torch

    from sqltuner.grpo import advantages, grpo_loss

    adv, zero = advantages(np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]), np.repeat([0, 1], 4))
    assert zero == 0.5 and np.allclose(adv[4:], 0) and adv[0] > 0 > adv[1] and abs(adv[:4].sum()) < 1e-9
    lp = torch.log(torch.tensor([[0.5, 0.25], [0.5, 0.25]]))
    a = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 2)
    per, kl = grpo_loss(lp, None, lp.clone(), a, mask, beta=0.1, clip_eps=0.2)
    assert torch.allclose(per, torch.tensor([[-1.0, -1.0], [1.0, 1.0]])) and torch.all(kl == 0)
    old = lp - 0.5  # ratio e^0.5 ≈ 1.65 is clipped at 1.2 for positive advantages
    per, _ = grpo_loss(lp, old, None, a, mask, beta=0.0, clip_eps=0.2)
    assert torch.allclose(per[0], torch.tensor([-1.2, -1.2]), atol=1e-6)
    ref = lp + 0.3
    _, kl = grpo_loss(lp, None, ref, a, mask, beta=0.1, clip_eps=0.2)
    assert torch.all(kl > 0)


def test_sampling_ignores_the_models_generation_config(tiny):
    """Qwen2.5-Coder's generation_config carries repetition_penalty 1.05–1.1, top_k and top_p: greedy must still be argmax and
    GRPO's samples must follow exactly the probabilities that token_logprobs computes."""
    import torch

    from sqltuner.generate import sampling_kwargs

    tok, model, _ = load_model(str(tiny), device="cpu")
    for k, v in {"repetition_penalty": 1.3, "top_k": 3, "top_p": 0.5, "temperature": 0.6, "do_sample": True}.items():
        setattr(model.generation_config, k, v)
    enc = tok([apply_chat_template(tok, [{"role": "user", "content": "Question one two three"}])], return_tensors="pt", add_special_tokens=False)
    P = enc["input_ids"].shape[1]
    for t in (0.0, 0.7, 1.0):
        torch.manual_seed(0)
        out = model.generate(**enc, **sampling_kwargs(t), max_new_tokens=6, output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id,
                             eos_token_id=None)
        with torch.no_grad():
            raw = model(out.sequences).logits[0, P - 1 : -1].float() / (t or 1.0)
        got = torch.stack([sc[0].float() for sc in out.scores])
        assert torch.allclose(torch.log_softmax(raw, -1), torch.log_softmax(got, -1), atol=1e-4), t


def test_logprobs_match_generate_and_update_direction(tiny):
    import torch

    from sqltuner.grpo import Samples, advantages, grpo_loss, grpo_step, sample_groups, token_logprobs
    from sqltuner.sft import add_lora

    tok, model, _ = load_model(str(tiny), device="cpu")
    tok.padding_side = "left"
    prompts = [apply_chat_template(tok, [{"role": "user", "content": "Question one two three"}]), apply_chat_template(tok, [{"role": "user", "content": "Q"}])]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    torch.manual_seed(1)
    out = model.generate(**enc, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=5, num_return_sequences=2, output_scores=True,
                         return_dict_in_generate=True, pad_token_id=tok.pad_token_id, eos_token_id=None)
    P = enc["input_ids"].shape[1]
    comp = out.sequences[:, P:]
    gen_lp = torch.stack([torch.log_softmax(sc.float(), -1).gather(-1, comp[:, t : t + 1]).squeeze(-1) for t, sc in enumerate(out.scores)], 1)
    seq = torch.cat([enc["input_ids"].repeat_interleave(2, 0), comp], 1)
    attn = torch.cat([enc["attention_mask"].repeat_interleave(2, 0), torch.ones_like(comp)], 1)
    s = Samples(seq, attn, (attn.cumsum(1) - 1).clamp(min=0), torch.ones_like(comp), [], np.repeat([0, 1], 2), comp.shape[1], 1.0)
    with torch.no_grad():
        assert float((token_logprobs(model, s, slice(0, 4), None) - gen_lp).abs().max()) < 1e-4  # left padding + positions are right

    tok, model, _ = load_model(str(tiny), device="cpu", trainable=True)
    model = add_lora(model, replace(LORA, r=8, alpha=16, dropout=0.0))
    torch.manual_seed(0)
    s = sample_groups(model, tok, prompts, 8, 1.0, 6, "cpu")
    rewards = np.random.default_rng(0).random(len(s.texts))
    adv_np, _ = advantages(rewards, s.group)
    adv, mask = torch.tensor(adv_np, dtype=torch.float32), s.comp_mask.float()
    model.train()
    with torch.no_grad():
        before = (token_logprobs(model, s, slice(0, 16), None) * mask).sum(1)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    for _ in range(3):
        opt.zero_grad()
        per, _ = grpo_loss(token_logprobs(model, s, slice(0, 16), None), None, None, adv, mask, 0.0, 0.2)
        (per.sum() / mask.sum()).backward()
        opt.step()
    with torch.no_grad():
        after = (token_logprobs(model, s, slice(0, 16), None) * mask).sum(1)
    assert np.corrcoef(adv_np, (after - before).numpy())[0, 1] > 0.5  # positive-advantage samples become more likely

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    stats = grpo_step(model, tok, prompts, lambda i, t: (float("S" in t), {"correct": "S" in t, "flags": ("x",) if i == 0 else ()}), opt,
                      replace(GRPO, group_size=4, max_new_tokens=6, micro_batch=3, beta=0.05), "cpu")
    assert {"reward", "zero_var_groups", "kl", "loss", "gen_s", "train_s", "flags"} <= set(stats) and stats["kl"] >= 0
