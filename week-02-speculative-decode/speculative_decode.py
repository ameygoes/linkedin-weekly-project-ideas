"""
Week 02 — Speculative Decoding
@AccidentalDataEngineer | Wednesday Skill Build

Draft model generates K tokens cheaply.
Target model validates all K in one forward pass.
Accepted tokens are guaranteed to match target distribution.

Result: 2-3x throughput improvement with zero quality loss.
"""
from __future__ import annotations

import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class SpecDecodeConfig:
    draft_model_id: str = "facebook/opt-125m"   # small, fast
    target_model_id: str = "facebook/opt-1.3b"  # large, slow
    k: int = 5                                   # tokens to speculate per round
    max_new_tokens: int = 200
    temperature: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DecodeStats:
    tokens_generated: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    target_forward_passes: int = 0
    draft_forward_passes: int = 0
    elapsed_s: float = 0.0
    accepted_per_round: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / self.draft_tokens_proposed

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_s == 0:
            return 0.0
        return self.tokens_generated / self.elapsed_s

    @property
    def speedup_vs_target_only(self) -> float:
        # if we'd used target for every token: tokens_generated target passes
        # we actually used: target_forward_passes
        if self.target_forward_passes == 0:
            return 0.0
        return self.tokens_generated / self.target_forward_passes

    def summary(self) -> str:
        return (
            f"tokens_generated:       {self.tokens_generated}\n"
            f"acceptance_rate:        {self.acceptance_rate:.1%}\n"
            f"target_forward_passes:  {self.target_forward_passes}\n"
            f"draft_forward_passes:   {self.draft_forward_passes}\n"
            f"tokens_per_second:      {self.tokens_per_second:.1f}\n"
            f"speedup_vs_target_only: {self.speedup_vs_target_only:.2f}x\n"
            f"elapsed_s:              {self.elapsed_s:.2f}"
        )


class SpeculativeDecoder:
    def __init__(self, cfg: SpecDecodeConfig):
        self.cfg = cfg
        print(f"Loading draft model: {cfg.draft_model_id}")
        self.draft_tok = AutoTokenizer.from_pretrained(cfg.draft_model_id)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            cfg.draft_model_id, torch_dtype=torch.float16
        ).to(cfg.device).eval()

        print(f"Loading target model: {cfg.target_model_id}")
        self.target_tok = AutoTokenizer.from_pretrained(cfg.target_model_id)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            cfg.target_model_id, torch_dtype=torch.float16
        ).to(cfg.device).eval()

    @torch.inference_mode()
    def _draft_k_tokens(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run draft model autoregressively for K steps.
        Returns (draft_ids [1, K], draft_probs [K, vocab])."""
        ids = input_ids.clone()
        draft_token_ids = []
        draft_probs = []

        for _ in range(self.cfg.k):
            out = self.draft_model(ids)
            logits = out.logits[:, -1, :]  # [1, vocab]
            probs = F.softmax(logits / self.cfg.temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)  # [1, 1]
            draft_token_ids.append(next_id)
            draft_probs.append(probs)
            ids = torch.cat([ids, next_id], dim=1)

        return (
            torch.cat(draft_token_ids, dim=1),       # [1, K]
            torch.cat(draft_probs, dim=0),           # [K, vocab]
        )

    @torch.inference_mode()
    def _target_verify(
        self, input_ids: torch.Tensor, draft_ids: torch.Tensor
    ) -> torch.Tensor:
        """Single target forward pass over prompt + K draft tokens.
        Returns target probs [K+1, vocab]."""
        full_ids = torch.cat([input_ids, draft_ids], dim=1)  # [1, N+K]
        out = self.target_model(full_ids)
        # logits for positions [N-1 .. N+K-1] → the K verification slots + 1 bonus
        logits = out.logits[0, -(self.cfg.k + 1) :, :]      # [K+1, vocab]
        return F.softmax(logits / self.cfg.temperature, dim=-1)

    @staticmethod
    def _rejection_sample(
        draft_probs: torch.Tensor,   # [K, vocab]
        target_probs: torch.Tensor,  # [K+1, vocab]
        draft_ids: torch.Tensor,     # [1, K]
    ) -> tuple[list[int], int]:
        """Accept/reject each draft token. Returns (accepted_ids, n_accepted)."""
        accepted = []
        k = draft_ids.shape[1]
        for i in range(k):
            token_id = draft_ids[0, i].item()
            p_draft  = draft_probs[i, token_id].item()
            p_target = target_probs[i, token_id].item()

            acceptance_prob = min(1.0, p_target / (p_draft + 1e-10))
            if torch.rand(1).item() < acceptance_prob:
                accepted.append(token_id)
            else:
                # reject: sample from corrected distribution
                corrected = torch.clamp(target_probs[i] - draft_probs[i], min=0)
                if corrected.sum() < 1e-8:
                    corrected = target_probs[i].clone()
                corrected = corrected / corrected.sum()
                fallback = torch.multinomial(corrected, 1).item()
                accepted.append(fallback)
                return accepted, len(accepted)

        # all K accepted — sample bonus token from target position K
        bonus = torch.multinomial(target_probs[k], 1).item()
        accepted.append(bonus)
        return accepted, k  # n_accepted = K (bonus is extra)

    def generate(self, prompt: str) -> tuple[str, DecodeStats]:
        stats = DecodeStats()
        tok = self.draft_tok  # assume same vocab (OPT family does)

        input_ids = tok(prompt, return_tensors="pt").input_ids.to(self.cfg.device)
        generated_ids = input_ids.clone()

        eos_id = tok.eos_token_id
        t0 = time.perf_counter()

        while stats.tokens_generated < self.cfg.max_new_tokens:
            # 1. Draft K tokens
            draft_ids, draft_probs = self._draft_k_tokens(generated_ids)
            stats.draft_forward_passes += self.cfg.k

            # 2. Target verifies in one pass
            target_probs = self._target_verify(generated_ids, draft_ids)
            stats.target_forward_passes += 1

            # 3. Rejection sampling
            accepted_ids, n_spec_accepted = self._rejection_sample(
                draft_probs, target_probs, draft_ids
            )

            stats.draft_tokens_proposed += self.cfg.k
            stats.draft_tokens_accepted += n_spec_accepted
            stats.accepted_per_round.append(n_spec_accepted)

            # 4. Append accepted tokens
            for tid in accepted_ids:
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[tid]], device=self.cfg.device)],
                    dim=1,
                )
                stats.tokens_generated += 1
                if tid == eos_id or stats.tokens_generated >= self.cfg.max_new_tokens:
                    break

            if generated_ids[0, -1].item() == eos_id:
                break

        stats.elapsed_s = time.perf_counter() - t0
        output_ids = generated_ids[0, input_ids.shape[1] :]
        return tok.decode(output_ids, skip_special_tokens=True), stats
