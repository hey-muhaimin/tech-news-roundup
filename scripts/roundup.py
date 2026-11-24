"""Hourly tech news & trends roundup.

Primary: Exa Answer API (search + LLM synthesis in one call, Exa's own model).
Fallback: Exa Search + OpenRouter free model (if Exa Answer is unavailable).

Runs in GitHub Actions (cron) — no local resources used.
"""

import os
import sys
import json
import datetime
from pathlib import Path

import requests

EXA_API_KEY = os.environ.get("EXA_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b-it:free").strip()

OUT_DIR = Path("roundups")


def exa_answer(query: str, system_prompt: str) -> str:
    """Exa Answer API: search + LLM synthesis in one call."""
    if not EXA_API_KEY:
        raise RuntimeError("EXA_API_KEY is not set")
    resp = requests.post(
        "https://api.exa.ai/answer",
        headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
        json={"query": query, "systemPrompt": system_prompt, "text": True},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("answer", "") or ""


def exa_search(query: str, num_results: int = 6) -> list[dict]:
    if not EXA_API_KEY:
        raise RuntimeError("EXA_API_KEY is not set")
    resp = requests.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
        json={
            "query": query,
            "numResults": num_results,
            "useAutoprompt": True,
            "contents": {"text": True},
        },
        timeout=60,
    )
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", []):
        out.append({"title": r.get("title", ""), "url": r.get("url", ""),
                    "text": (r.get("text") or "")[:800]})
    return out


def openrouter_chat(messages: list[dict]) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/tech-news-roundup",
            "X-Title": "Tech News Roundup",
        },
        json={"model": OPENROUTER_MODEL, "messages": messages, "max_tokens": 1200, "temperature": 0.4},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


SYSTEM_PROMPT = (
    "You are a concise tech news editor. Produce a Markdown roundup with: "
    "(1) Top 10 stories with a one-line takeaway each, grouped by theme "
    "(AI, funding, security, hardware, geopolitics, consumer); "
    "(2) Five emerging tech trends this week; "
    "(3) a short Watch List of 3 items. Keep under 900 words. "
    "Cite source URLs inline. No code blocks."
)


def build_via_exa_answer() -> str:
    query = "What are the top technology news stories and emerging tech trends today?"
    return exa_answer(query, SYSTEM_PROMPT)


def build_via_search_plus_llm() -> str:
    queries = [
        "top technology news today",
        "latest AI startup funding and product launches",
        "cybersecurity news this week",
        "emerging tech trends 2026",
    ]
    chunks = []
    for q in queries:
        try:
            for r in exa_search(q, num_results=5):
                chunks.append(f"- **{r['title']}** ({r['url']}) {r['text'][:400]}")
        except Exception as e:
            print(f"[warn] exa search failed for {q!r}: {e}", file=sys.stderr)
    research = "\n".join(chunks)
    if not research.strip():
        raise RuntimeError("No research gathered")
    return openrouter_chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Research snippets:\n\n{research}\n\nWrite the roundup now."},
    ])


def main() -> int:
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime("%Y-%m-%d-%H")
    path = OUT_DIR / f"tech-news-roundup-{stamp}.md"

    md = None
    # Primary: Exa Answer (search + synthesis, single call, no rate-limit thrash)
    try:
        md = build_via_exa_answer()
        print("Used Exa Answer API")
    except Exception as e:
        print(f"[warn] Exa Answer failed ({e}); falling back to search + OpenRouter", file=sys.stderr)
        try:
            md = build_via_search_plus_llm()
            print("Used Exa Search + OpenRouter")
        except Exception as e2:
            print(f"[error] fallback failed: {e2}", file=sys.stderr)
            return 1

    if not md or not md.strip():
        print("Empty result; aborting.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    header = f"# Tech News & Trends Roundup — {now.strftime('%Y-%m-%d %H:00 UTC')}\n\n"
    path.write_text(header + md.strip() + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
