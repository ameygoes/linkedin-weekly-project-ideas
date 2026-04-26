# Week 01 — KV Cache for Local LLM Inference

**Built:** Wednesday Apr 23, 2026  
**Stack:** llama.cpp + Redis + FastAPI  
**Result:** 1.8s → 0.4s on cache hits, 73% hit rate

---

## What this is

A KV cache layer sitting in front of a local LLM (llama.cpp) backed by Redis.

The DE analogy: it's a materialised view for attention.  
Compute the Key/Value matrices once for a prompt prefix. Cache them. Serve repeated prompts without recomputing.

---

## The bug that took 90 minutes

Redis eviction policy. `maxmemory-policy noeviction` (default) caused silent cache misses under memory pressure. The cache appeared to work — it was silently dropping keys.

Fix: one line in `redis.conf`:

```
maxmemory-policy allkeys-lru
```

---

## Quickstart

```bash
# 1. Start Redis
redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru

# 2. Start llama.cpp server (replace model path)
./llama-server -m models/mistral-7b-q4.gguf --port 8080

# 3. Install deps
pip install fastapi uvicorn redis httpx

# 4. Run the cache server
uvicorn server:app --port 9000

# 5. Test it
python test_cache.py
```

---

## Results

```
latency_p50 before cache:  1,847ms
latency_p50 after  cache:    412ms
cache hit rate:               73%
tokens saved (1 day test):  41,200
estimated cost saved vs API: ~$0.82/day
```
