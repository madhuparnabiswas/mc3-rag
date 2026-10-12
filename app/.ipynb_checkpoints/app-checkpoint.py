#!/usr/bin/env python3
"""Reference submission skeleton for AMD Mini-Challenge 3 (RAG).

Unlike MC-2, the harness invokes this script in TWO different ways, inside your
already-running container.

1. ONCE, before any graded clock starts, to build whatever index you want:

       python3 /app/app.py --index /app/corpus

   This pass is ungraded and charged to your startup budget, not to a query.
   It is where you should pay for parsing PDFs, spreadsheets and images.

2. ONCE PER QUESTION, ten times:

       python3 /app/app.py --corpus /app/corpus \\
                           --query-id query_01 \\
                           --query "What is the maximum junction temperature?"

   and then it reads the file you wrote:

       /app/output/query_01_output.json

   The query id is GIVEN to you, not derived -- a question has no filename to
   take a stem from. Write to exactly that id plus _output.json.

The output is a JSON object with three keys:

    {"answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.9}

Replace index() and answer() below. Leave the plumbing alone.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
SERVER_URL = "http://127.0.0.1:8000"


def ensure_server_running(timeout: float = 120.0) -> None:
    """Checks if server.py is running. If not, spawns it in the background."""
    deadline = time.time() + timeout

    # 1. Check if already responding
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=0.5) as resp:
            if resp.status == 200:
                return
    except Exception:
        pass

    # 2. If not running, start server.py in the background
    print(
        "[App] Background server not running. Spawning server.py...", flush=True
    )
    subprocess.Popen(
        [sys.executable, "/app/server.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # 3. Wait for it to load models and report healthy
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=1.0) as resp:
                if resp.status == 200:
                    print("[App] Background server is active.", flush=True)
                    return
        except Exception:
            time.sleep(0.5)

    raise RuntimeError("Timed out waiting for background server to boot.")


def send_to_server(
    endpoint: str, payload: dict, timeout: float = 30.0
) -> dict:
    ensure_server_running()
    url = f"{SERVER_URL}/{endpoint}"
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def index(corpus: Path) -> None:
    """Delegates multi-modal document extraction & indexing to the Qwen2.5-VL server."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if not corpus.exists() or not corpus.is_dir():
        print(f"[App] Corpus path {corpus} does not exist or is not a directory.", flush=True)
        return

    try:
        # High timeout for VLM extraction across PDFs and images during the ungraded index pass
        res = send_to_server(
            "index",
            {"corpus": str(corpus.resolve())},
            timeout=600.0,
        )
        print(
            f"[App] Indexing complete. Successfully processed {res.get('count', 0)} files.",
            flush=True,
        )
    except Exception as e:
        print(f"[App] Error during indexing via Qwen server: {e}", flush=True)


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    """Queries the resident Qwen2.5-VL server to answer the question from indexed docs."""
    try:
        res = send_to_server(
            "query",
            {
                "corpus": str(corpus.resolve()),
                "query": query,
            },
            timeout=25.0,
        )
        ans = str(res.get("answer", "")).strip()
        citations = [
            str(c).replace("\\", "/").strip()
            for c in res.get("citations", [])
            if c
        ]
        confidence = float(res.get("confidence", 0.0))

        if not ans or not citations:
            return "", [], 0.0

        return ans, citations, confidence
    except Exception as e:
        print(f"[App] Error querying Qwen server: {e}", flush=True)
        return "", [], 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    text, citations, confidence = answer(args.corpus, args.query)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (args.query_id + "_output.json")

    
    out.write_text(
        json.dumps(
            {"answer": text, "citations": list(citations), "confidence": confidence},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())