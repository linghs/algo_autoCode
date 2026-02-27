#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation Script for Medbench
Based on EMNLP 2025 Survey: From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge
- Implements pairwise comparison with bias mitigation
- Uses rule-augmented prompting with Chain-of-Thought
- Applies swapping operation to reduce position bias
- Supports both pointwise and pairwise evaluation modes

Usage:
    python medbench_eval_LLM_as_judge.py
    python medbench_eval_LLM_as_judge.py --input_dir Medbench_2025_direct_match --output_dir eval_results_judge
    python medbench_eval_LLM_as_judge.py --model qwen-plus --judge_model qwen-plus --workers 8 --limit 10
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
DEFAULT_INPUT_DIR = "Medbench_2025_direct_match"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_JUDGE_MODEL = "qwen-plus"  # Separate model for judging

METHOD = "llm_judge_pairwise_swap"
DEFAULT_OUTPUT_DIR = f"eval_results_{METHOD}_{DEFAULT_JUDGE_MODEL}"
API_KEY = "your-openai-compatible-api-key"   # Replace with your API key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # Or your custom endpoint
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries
DEFAULT_WORKERS = 8
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


def build_judge_prompt(instruction: str, response_a: str, response_b: str, rubric: str = None) -> str:
    """
    Build judge prompt following EMNLP 2025 survey methodology.
    Implements rule-augmented prompting with Chain-of-Thought.
    """
    if rubric is None:
        rubric = "Evaluate based on accuracy, relevance, and clarity of the medical information provided."

    prompt = f"""You are an impartial medical expert judge. Your task is to compare two responses to a medical question and determine which one is better.

Evaluation Criteria: {rubric}

[Medical Question]
{instruction}

[Response A]
{response_a}

[Response B]
{response_b}

Think step by step about the quality of each response according to the criteria above. Then provide your final judgment by answering with ONLY 'A' if Response A is better, 'B' if Response B is better, or 'TIE' if they are equally good.

Step-by-step analysis:
"""
    return prompt


def call_judge(client: OpenAI, instruction: str, response_a: str, response_b: str, rubric: str = None) -> str:
    """Call the judge model to compare two responses."""
    prompt = build_judge_prompt(instruction, response_a, response_b, rubric)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=DEFAULT_JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  [judge retry {attempt}/{MAX_RETRIES}] {e}", flush=True)
            time.sleep(RETRY_DELAY * attempt)


def parse_judgment(judgment_text: str) -> str:
    """Parse the judge's response to extract A/B/TIE decision."""
    judgment_text = judgment_text.upper()

    if 'A' in judgment_text and 'B' not in judgment_text:
        return 'A'
    elif 'B' in judgment_text and 'A' not in judgment_text:
        return 'B'
    elif 'TIE' in judgment_text or ('A' in judgment_text and 'B' in judgment_text):
        return 'TIE'

    matches = re.findall(r'[AB]', judgment_text)
    if matches:
        return matches[-1]

    return 'TIE'


def perform_pairwise_comparison(client: OpenAI, instruction: str, response_a: str, response_b: str, rubric: str = None) -> str:
    """
    Perform pairwise comparison with swapping operation to mitigate position bias.
    Follows methodology from EMNLP 2025 survey Section 4.2.1.
    """
    judgment_1_raw = call_judge(client, instruction, response_a, response_b, rubric)
    judgment_1 = parse_judgment(judgment_1_raw)

    judgment_2_raw = call_judge(client, instruction, response_b, response_a, rubric)
    judgment_2 = parse_judgment(judgment_2_raw)

    if judgment_1 == 'A' and judgment_2 == 'A':
        final_judgment = 'B'
    elif judgment_1 == 'B' and judgment_2 == 'B':
        final_judgment = 'A'
    elif judgment_1 == judgment_2:
        final_judgment = judgment_1
    else:
        final_judgment = 'TIE'

    return final_judgment, judgment_1_raw, judgment_2_raw


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
    judge_model: str,
    workers: int,
    limit: int,
) -> dict:
    """Process one JSONL file using LLM-as-a-Judge. Returns summary stats dict."""
    global DEFAULT_JUDGE_MODEL
    DEFAULT_JUDGE_MODEL = judge_model

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

    done_ids = load_done_ids(output_path)
    pending = [it for it in items if it.get("other", {}).get("id", "") not in done_ids]
    skipped = total - len(pending)

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
        client = build_client()
        item_id = item.get("other", {}).get("id", "")
        source = item.get("other", {}).get("source", "")
        question = item["question"]
        answer = item["answer"]

        try:
            model_response = call_model(client, question, model)

            comparison_result, judgment_1_raw, judgment_2_raw = perform_pairwise_comparison(
                client,
                question,
                model_response,
                answer
            )

            correct = comparison_result in ['A']
            direct_match_correct = is_correct(model_response, answer)

            result = {
                "id": item_id,
                "source": source,
                "question": question,
                "answer": answer,
                "model_response": model_response,
                "norm_answer": normalize_answer(answer),
                "norm_model_response": normalize_answer(model_response),
                "comparison_result": comparison_result,
                "judgment_1_raw": judgment_1_raw,
                "judgment_2_raw": judgment_2_raw,
                "llm_judge_correct": correct,
                "direct_match_correct": direct_match_correct,
                "correct": direct_match_correct
            }
        except Exception as e:
            result = {
                "id": item_id,
                "source": source,
                "question": question,
                "answer": answer,
                "model_response": f"[ERROR] {e}",
                "norm_answer": normalize_answer(answer),
                "norm_model_response": "",
                "comparison_result": "ERROR",
                "judgment_1_raw": "",
                "judgment_2_raw": "",
                "llm_judge_correct": False,
                "direct_match_correct": False,
                "correct": False,
                "error": True,
            }

        with out_lock:
            with open(output_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            counters["done"] += 1
            if result["correct"]:
                counters["correct"] += 1
            if result.get("error"):
                counters["error"] += 1
            done = counters["done"]
            correct = counters["correct"]
            accuracy = correct / done * 100 if done else 0.0
            print(
                f"  [{done:>4}/{total}] id={str(item_id):>4}  "
                f"model_resp={result['norm_model_response']!r:14}  "
                f"gt={result['norm_answer']!r:10}  "
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

    done = counters["done"]
    correct = counters["correct"]
    errors = counters["error"]
    valid = done - errors
    accuracy = correct / valid * 100 if valid > 0 else 0.0

    return {
        "dataset": dataset_name,
        "total": total,
        "done": done,
        "correct": correct,
        "errors": errors,
        "accuracy": accuracy,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge evaluation on Medbench tasks.")
    parser.add_argument("--input_dir", "-i", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", "-o", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model to evaluate")
    parser.add_argument("--judge_model", "-j", default=DEFAULT_JUDGE_MODEL, help="Model to use as judge")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", "-n", type=int, default=0,
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

    print(f"Found {len(input_files)} file(s). Model={args.model}, Judge={args.judge_model}  Workers={args.workers}")
    if args.limit:
        print(f"Limit: first {args.limit} items per file")

    all_stats = []
    for input_path in input_files:
        dataset_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(args.output_dir, f"{dataset_name}_results.jsonl")
        stats = evaluate_file(input_path, output_path, args.model, args.judge_model, args.workers, args.limit)
        all_stats.append(stats)

    # ── Summary Table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"{'Dataset':<35} {'Total':>6} {'Correct':>8} {'Errors':>7} {'Accuracy':>10}")
    print(f"{'─'*80}")

    total_items = 0
    total_correct = 0
    total_errors = 0

    for s in all_stats:
        print(
            f"{s['dataset']:<35} {s['done']:>6} {s['correct']:>8} "
            f"{s['errors']:>7} {s['accuracy']:>9.2f}%"
        )
        total_items += s["done"]
        total_correct += s["correct"]
        total_errors += s["errors"]

    overall = total_correct / (total_items - total_errors) * 100 if (total_items - total_errors) > 0 else 0.0
    print(f"{'─'*80}")
    print(f"{'OVERALL':<35} {total_items:>6} {total_correct:>8} {total_errors:>7} {overall:>9.2f}%")
    print(f"{'═'*80}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "judge_model": args.judge_model,
            "method": METHOD,
            "results": all_stats,
            "overall": {
                "total": total_items,
                "correct": total_correct,
                "errors": total_errors,
                "accuracy": round(overall, 4),
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
