"""Quick benchmark — run twice to see cache hit latency vs miss."""

import time
import httpx

SERVER = "http://localhost:9000"
PROMPT = "Explain KV cache in LLMs using a data engineering analogy."


def request(prompt: str) -> dict:
    t0 = time.perf_counter()
    r = httpx.post(f"{SERVER}/completion", json={"prompt": prompt, "max_tokens": 128}, timeout=120)
    elapsed = (time.perf_counter() - t0) * 1000
    data = r.json()
    return {"latency_ms": round(elapsed, 1), "cache_hit": data.get("cache_hit"), "content": data.get("content", "")[:80]}


if __name__ == "__main__":
    print("Run 1 (cache miss expected):")
    print(request(PROMPT))

    print("\nRun 2 (cache hit expected):")
    print(request(PROMPT))

    print("\nStats:")
    print(httpx.get(f"{SERVER}/stats").json())
