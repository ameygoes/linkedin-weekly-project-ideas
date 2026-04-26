"""
KV Cache proxy for llama.cpp — FastAPI + Redis backend.
Built on a Wednesday. Code is real, not polished.
"""

import hashlib
import json

import httpx
import redis
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="llama-kv-cache")

# Redis client — make sure maxmemory-policy allkeys-lru is set
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

LLAMA_URL = "http://localhost:8080/completion"
CACHE_TTL = 3600  # seconds


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


def cache_key(req: CompletionRequest) -> str:
    payload = f"{req.prompt}|{req.max_tokens}|{req.temperature}"
    return "kvcache:" + hashlib.sha256(payload.encode()).hexdigest()


@app.post("/completion")
async def completion(req: CompletionRequest):
    key = cache_key(req)

    # Cache hit
    cached = r.get(key)
    if cached:
        return {"content": json.loads(cached), "cache_hit": True}

    # Cache miss — call llama.cpp
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(LLAMA_URL, json={
            "prompt": req.prompt,
            "n_predict": req.max_tokens,
            "temperature": req.temperature,
        })
    resp.raise_for_status()
    content = resp.json().get("content", "")

    # Store in Redis
    r.setex(key, CACHE_TTL, json.dumps(content))

    return {"content": content, "cache_hit": False}


@app.get("/stats")
def stats():
    info = r.info("stats")
    return {
        "cache_hits": info.get("keyspace_hits", 0),
        "cache_misses": info.get("keyspace_misses", 0),
        "hit_rate": round(
            info.get("keyspace_hits", 0) /
            max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100,
            1
        ),
        "total_keys": r.dbsize(),
    }
