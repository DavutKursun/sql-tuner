"""A tiny random causal LM with a real byte-level BPE tokenizer and a chat template, built offline.

Tests and smoke runs use it so that every code path (chat template, label masking, LoRA, generation, GRPO) runs with
no download and no GPU. It learns nothing useful about SQL; that is not its job.
"""

from __future__ import annotations

from pathlib import Path

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)
SPECIAL = ["<|pad|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]
CORPUS = [
    "SELECT name FROM singer WHERE age > 30", "SELECT count(*) FROM concert", "select T1.name from a as T1 join b as T2 on T1.id = T2.id",
    "Database schema:", "CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);", "Question: How many singers are there?",
    "```sql\nSELECT count(*) FROM singer\n```", "ORDER BY GROUP BY HAVING LIMIT DISTINCT AVG MAX MIN SUM UNION INTERSECT EXCEPT",
    "You are an expert SQLite analyst.", "example rows", "the quick brown fox 0123456789 | ' \" ( ) , ; * = < > _",
]


def build_tiny(out_dir: Path, vocab_size: int = 400, hidden: int = 64, layers: int = 2, heads: int = 4, seed: int = 0) -> Path:
    """Write a tokenizer + random LlamaForCausalLM to out_dir; returns out_dir (loadable with from_pretrained)."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(CORPUS * 20, trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<|pad|>", eos_token="<|end|>", bos_token=None, unk_token=None,
                                   model_input_names=["input_ids", "attention_mask"])  # like Qwen's: no token_type_ids
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(out_dir)
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(fast), hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=heads, max_position_embeddings=4096, pad_token_id=fast.pad_token_id, eos_token_id=fast.eos_token_id,
                      tie_word_embeddings=True, initializer_range=0.2)  # large enough that a frozen embedding can still give confident logits
    model = LlamaForCausalLM(cfg)
    model.generation_config.pad_token_id = fast.pad_token_id
    model.generation_config.eos_token_id = fast.eos_token_id
    model.save_pretrained(out_dir)
    return out_dir
