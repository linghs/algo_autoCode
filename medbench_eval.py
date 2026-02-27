#!/usr/bin/env python3
"""
Medbench Direct-Match Evaluation Script
- Reads all JSONL files from Medbench_2025_direct_match/
- Calls the OpenAI-compatible API for each question
- Compares predictions with ground truth (direct match with normalization)
- Saves per-file results and prints a summary table

Usage:
    python medbench_eval.py
    python medbench_eval.py --input_dir Medbench_2025_direct_match --output_dir eval_results
    python medbench_eval.py --model qwen-plus --workers 8 --limit 10
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR  = "Medbench_2025_direct_match"
DEFAULT_MODEL      = "qwen-plus"

METHOD         = ""
DEFAULT_OUTPUT_DIR = f"eval_results_{METHOD}_{DEFAULT_MODEL}"
API_KEY            = "your-qwen-compatible-api-key"   # Replace with your API key
BASE_URL           = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # Or your custom endpoint
MAX_RETRIES        = 3
RETRY_DELAY        = 2   # seconds between retries
DEFAULT_WORKERS    = 8
# ──────────────────────────────────────────────────────────────────────────────


def build_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_model(client: OpenAI, question: str, model: str) -> str:
    """Call the LLM and return the raw response text."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0,
                max_tokens=128,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [retry {attempt}/{MAX_RETRIES}] {e}", flush=True)
            time.sleep(RETRY_DELAY * attempt)


# ── Answer Normalization ───────────────────────────────────────────────────────

def _extract_angle_bracket(text: str) -> str:
    """Extract content inside the last <…> pair, or return text as-is."""
    matches = re.findall(r"<([^<>]+)>", text)
    return matches[-1] if matches else text


def normalize_answer(text: str) -> str:
    """
    Normalize an answer for comparison:
    - Strip surrounding whitespace
    - If it contains <…>, extract the inner content, split by comma,
      strip each option, sort, and rejoin as '<A,B,C>'
    - If it contains '|', split, strip each item, sort, rejoin as 'X|Y'
    - Otherwise, strip and lowercase
    """
    text = text.strip()

    if "<" in text and ">" in text:
        inner = _extract_angle_bracket(text)
        options = sorted(o.strip() for o in inner.split(",") if o.strip())
        return "<" + ",".join(options) + ">"

    if "|" in text:
        parts = sorted(p.strip() for p in text.split("|") if p.strip())
        return "|".join(parts)

    return text.strip()


def is_correct(prediction: str, answer: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(answer)


# ── Resume Support ─────────────────────────────────────────────────────────────

def load_done_ids(output_path: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ── Per-file Evaluation ────────────────────────────────────────────────────────

def evaluate_file(
    input_path: str,
    output_path: str,
    model: str,
    workers: int,
    limit: int,
) -> dict:
    """Process one JSONL file. Returns summary stats dict."""
    dataset_name = os.path.splitext(os.path.basename(input_path))[0]

    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if limit > 0:
        items = items[:limit]

    total = len(items)

    done_ids    = load_done_ids(output_path)
    pending     = [it for it in items if it.get("other", {}).get("id", "") not in done_ids]
    skipped     = total - len(pending)

    print(f"\n{'─'*60}")
    print(f"[{dataset_name}]  total={total}  pending={len(pending)}  skipped={skipped}")

    out_lock = Lock()
    counters = {"done": skipped, "correct": 0, "error": 0}

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if obj.get("correct"):
                        counters["correct"] += 1
                except (json.JSONDecodeError, KeyError):
                    pass

    def process_and_save(item: dict) -> None:
        client    = build_client()
        item_id   = item.get("other", {}).get("id", "")
        source    = item.get("other", {}).get("source", "")
        question  = item["question"]
        answer    = item["answer"]

        try:
            prediction = call_model(client, question, model)
            correct    = is_correct(prediction, answer)
            result = {
                "id":         item_id,
                "source":     source,
                "question":   question,
                "answer":     answer,
                "prediction": prediction,
                "norm_answer":     normalize_answer(answer),
                "norm_prediction": normalize_answer(prediction),
                "correct":    correct,
            }
        except Exception as e:
            result = {
                "id":         item_id,
                "source":     source,
                "question":   question,
                "answer":     answer,
                "prediction": f"[ERROR] {e}",
                "norm_answer":     normalize_answer(answer),
                "norm_prediction": "",
                "correct":    False,
                "error":      True,
            }

        with out_lock:
            with open(output_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            counters["done"] += 1
            if result["correct"]:
                counters["correct"] += 1
            if result.get("error"):
                counters["error"] += 1
            done     = counters["done"]
            correct  = counters["correct"]
            accuracy = correct / done * 100 if done else 0.0
            print(
                f"  [{done:>4}/{total}] id={str(item_id):>4}  "
                f"norm_pred={result['norm_prediction']!r:14}  "
                f"norm_ans={result['norm_answer']!r:10}  "
                f"{'✓' if result['correct'] else '✗'}  acc={accuracy:.1f}%",
                flush=True,
            )

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_and_save, it): it for it in pending}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    print(f"  [Unhandled] {exc}", flush=True)

    done     = counters["done"]
    correct  = counters["correct"]
    errors   = counters["error"]
    valid    = done - errors
    accuracy = correct / valid * 100 if valid > 0 else 0.0

    return {
        "dataset":  dataset_name,
        "total":    total,
        "done":     done,
        "correct":  correct,
        "errors":   errors,
        "accuracy": accuracy,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch evaluation on Medbench direct-match tasks.")
    parser.add_argument("--input_dir",  "-i", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", "-o", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model",      "-m", default=DEFAULT_MODEL)
    parser.add_argument("--workers",    "-w", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit",      "-n", type=int, default=0,
                        help="Process only first N items per file (0 = all)")
    parser.add_argument("--files", nargs="*",
                        help="Specific file names to run (default: all .jsonl in input_dir)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[Error] Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.files:
        input_files = [os.path.join(args.input_dir, f) for f in args.files]
    else:
        input_files = sorted(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".jsonl")
        )

    if not input_files:
        print("[Error] No .jsonl files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(input_files)} file(s). Model={args.model}  Workers={args.workers}")
    if args.limit:
        print(f"Limit: first {args.limit} items per file")

    all_stats = []
    for input_path in input_files:
        dataset_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path  = os.path.join(args.output_dir, f"{dataset_name}_results.jsonl")
        stats = evaluate_file(input_path, output_path, args.model, args.workers, args.limit)
        all_stats.append(stats)

    # ── Summary Table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"{'Dataset':<35} {'Total':>6} {'Correct':>8} {'Errors':>7} {'Accuracy':>10}")
    print(f"{'─'*70}")

    total_items   = 0
    total_correct = 0
    total_errors  = 0

    for s in all_stats:
        print(
            f"{s['dataset']:<35} {s['done']:>6} {s['correct']:>8} "
            f"{s['errors']:>7} {s['accuracy']:>9.2f}%"
        )
        total_items   += s["done"]
        total_correct += s["correct"]
        total_errors  += s["errors"]

    overall = total_correct / (total_items - total_errors) * 100 if (total_items - total_errors) > 0 else 0.0
    print(f"{'─'*70}")
    print(f"{'OVERALL':<35} {total_items:>6} {total_correct:>8} {total_errors:>7} {overall:>9.2f}%")
    print(f"{'═'*70}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model":   args.model,
            "results": all_stats,
            "overall": {
                "total":    total_items,
                "correct":  total_correct,
                "errors":   total_errors,
                "accuracy": round(overall, 4),
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
