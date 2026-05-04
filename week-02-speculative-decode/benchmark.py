"""
benchmark.py — compare speculative decode vs target-only baseline.

Usage:
    python benchmark.py
    python benchmark.py --k 7 --max-new-tokens 150
    python benchmark.py --draft facebook/opt-125m --target facebook/opt-1.3b
"""
from __future__ import annotations

import argparse
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from speculative_decode import SpecDecodeConfig, SpeculativeDecoder

PROMPTS = [
    "Explain how a data lakehouse architecture differs from a traditional data warehouse in three paragraphs.",
    "Write a Python function that reads a Parquet file from S3 using PyArrow and filters rows where column 'status' equals 'active'.",
    "What are the main tradeoffs between Apache Kafka and Apache Pulsar for a real-time data pipeline?",
    "Describe the medallion architecture (bronze, silver, gold layers) and when you would choose it over a simpler flat model.",
]


def baseline_generate(
    model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: str
) -> tuple[str, float, int]:
    """Standard autoregressive generation — one target forward pass per token."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    new_ids = out[0, input_ids.shape[1]:]
    tokens_generated = new_ids.shape[0]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, elapsed, tokens_generated


def run_benchmark(args: argparse.Namespace) -> None:
    cfg = SpecDecodeConfig(
        draft_model_id=args.draft,
        target_model_id=args.target,
        k=args.k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"\n{'='*60}")
    print(f"SPECULATIVE DECODING BENCHMARK")
    print(f"  draft:          {cfg.draft_model_id}")
    print(f"  target:         {cfg.target_model_id}")
    print(f"  k (speculate):  {cfg.k}")
    print(f"  max_new_tokens: {cfg.max_new_tokens}")
    print(f"  device:         {cfg.device}")
    print(f"{'='*60}\n")

    # Load target separately for baseline
    print("Loading target model for baseline...")
    target_tok = AutoTokenizer.from_pretrained(cfg.target_model_id)
    target_model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model_id, torch_dtype=torch.float16
    ).to(cfg.device).eval()

    decoder = SpeculativeDecoder(cfg)

    results = []
    for i, prompt in enumerate(PROMPTS[: args.num_prompts]):
        print(f"\n--- Prompt {i+1}/{args.num_prompts} ---")
        print(f"  {prompt[:80]}...")

        # Baseline
        _, b_elapsed, b_tokens = baseline_generate(
            target_model, target_tok, prompt,
            cfg.max_new_tokens, cfg.temperature, cfg.device,
        )
        b_tps = b_tokens / b_elapsed

        # Speculative
        _, stats = decoder.generate(prompt)
        s_tps = stats.tokens_per_second

        speedup = s_tps / b_tps if b_tps > 0 else 0
        results.append({
            "baseline_tps": b_tps,
            "spec_tps": s_tps,
            "speedup": speedup,
            "acceptance_rate": stats.acceptance_rate,
            "avg_tokens_per_round": (
                stats.tokens_generated / max(1, stats.target_forward_passes)
            ),
        })

        print(f"  baseline:   {b_tps:.1f} tok/s  ({b_elapsed:.2f}s, {b_tokens} tokens)")
        print(f"  spec_decode:{s_tps:.1f} tok/s  ({stats.elapsed_s:.2f}s, {stats.tokens_generated} tokens)")
        print(f"  speedup:    {speedup:.2f}x")
        print(f"  acceptance: {stats.acceptance_rate:.1%}")
        print(f"  avg accepted/round: {results[-1]['avg_tokens_per_round']:.2f}")

    # Summary
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    avg_acceptance = sum(r["acceptance_rate"] for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(results)} prompts)")
    print(f"  avg speedup:     {avg_speedup:.2f}x")
    print(f"  avg acceptance:  {avg_acceptance:.1%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft",  default="facebook/opt-125m")
    parser.add_argument("--target", default="facebook/opt-1.3b")
    parser.add_argument("--k",      type=int,   default=5)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--num-prompts",    type=int, default=4)
    args = parser.parse_args()
    run_benchmark(args)
