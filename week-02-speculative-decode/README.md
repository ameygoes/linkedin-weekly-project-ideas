# Week 02 — Speculative Decoding

**Wednesday Skill Build | @AccidentalDataEngineer**

Draft model proposes K tokens. Target model verifies in one forward pass.
Accepted tokens are statistically identical to target-only output. 2-3x faster.

## How it works

```
loop:
  1. draft model generates [t1..t5] cheaply (5 small forward passes)
  2. target model validates all 5 in ONE forward pass
  3. rejection sampling: accept ti with prob min(1, p_target/p_draft)
  4. on first rejection: resample from corrected distribution, discard rest
  5. if all accepted: sample one bonus token from target for free
```

The output distribution is provably identical to running target alone.
Speedup comes from draft acceptance rate × (draft_speed / target_speed).

## DE analogy

Speculative decoding = **query pushdown**.  
The draft model is a cheap local pre-filter that guesses results.  
The target engine validates. Right guess → skip the expensive scan.  
Wrong guess → fall back cleanly, nothing corrupted.

## Results (OPT-125m draft, OPT-1.3B target, K=5)

```
avg speedup:       2.1x
avg acceptance:    71%
avg tokens/round:  4.3 (out of 5 speculated)
```

## Usage

```bash
pip install -r requirements.txt

# Run benchmark (downloads models on first run, ~1.5GB)
python benchmark.py

# Custom models / K value
python benchmark.py --draft facebook/opt-125m --target facebook/opt-1.3b --k 7

# Use in code
from speculative_decode import SpecDecodeConfig, SpeculativeDecoder

cfg = SpecDecodeConfig(
    draft_model_id="facebook/opt-125m",
    target_model_id="facebook/opt-1.3b",
    k=5,
    max_new_tokens=200,
)
decoder = SpeculativeDecoder(cfg)
text, stats = decoder.generate("Explain data mesh architecture")
print(stats.summary())
```

## Files

| File | Purpose |
|------|---------|
| `speculative_decode.py` | Core implementation: draft, verify, rejection sampling |
| `benchmark.py` | Side-by-side latency comparison vs target-only baseline |
| `requirements.txt` | Dependencies |

## Key parameters

| Param | Default | Effect |
|-------|---------|--------|
| `k` | 5 | Tokens to speculate per round. Higher K = more speedup if acceptance is high, more waste if low |
| `temperature` | 1.0 | Affects acceptance rate: lower temp → higher acceptance (distributions more peaked) |

## What broke during the build

Redis eviction killed cache hits silently last week (KV Cache build).  
This week: first OPT model pairing had acceptance rate of 23% — useless.  
Fix: matched vocab size and temperature. OPT-125m → OPT-1.3B gives 71%.  
Draft and target models must share the same tokenizer vocabulary or  
the token ID mapping breaks the acceptance math entirely.

---

*Code is intentionally not clean. This is what Wednesday night looks like.*
