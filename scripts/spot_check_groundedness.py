"""Human spot-check of the LLM groundedness judge — an independent check on
whether claude-haiku-4-5's grounded/ungrounded calls actually agree with a
human reading the same question, answer, and cited source text.

An LLM (even a different, larger model) reviewing another LLM's judgments
doesn't test what this is for — the point is an independent human read.
This script sources the sample and records answers; a human runs it.

Usage:
  uv run python scripts/spot_check_groundedness.py            # run the review
  uv run python scripts/spot_check_groundedness.py --summarize # score agreement
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

EVAL_RESULTS_DIR = Path("eval_results")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
OUTPUT_PATH = EVAL_RESULTS_DIR / "human_spot_check.json"
SAMPLE_FRACTION = 0.20
SAMPLE_SEED = 42  # fixed, so re-running without a saved file reproduces the same sample


def _load_chunks_by_id() -> dict[str, dict]:
    chunks_by_id = {}
    with CHUNKS_PATH.open() as f:
        for line in f:
            chunk = json.loads(line)
            chunks_by_id[chunk["chunk_id"]] = chunk
    return chunks_by_id


def _load_judged_records() -> list[dict]:
    records = []
    for path in sorted(EVAL_RESULTS_DIR.glob("generation_*.json")):
        stage = path.stem.removeprefix("generation_")
        data = json.loads(path.read_text())
        for r in data["per_query"]:
            if r.get("grounded") is not None:
                records.append({**r, "stage": stage})
    return records


def _sample(records: list[dict], fraction: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    n = max(1, round(len(records) * fraction))
    return rng.sample(records, n)


def run_review() -> None:
    chunks_by_id = _load_chunks_by_id()
    records = _load_judged_records()
    sample = _sample(records, SAMPLE_FRACTION, SAMPLE_SEED)

    print(f"{len(records)} judged records across all stages; reviewing a {SAMPLE_FRACTION:.0%} "
          f"sample ({len(sample)} items, seed={SAMPLE_SEED}).\n")
    print("For each item: read the question, answer, and cited source text, then judge for "
          "yourself whether every factual claim in the answer is actually supported by the "
          "source. Enter y (grounded) / n (not grounded) / s (skip).\n")

    results = []
    for i, r in enumerate(sample, start=1):
        print(f"--- {i}/{len(sample)}  [{r['stage']} / {r['question_id']}] ---")
        print(f"Q: {r['question']}")
        print(f"A: {r['answer']}")
        print("Sources:")
        for cid in r.get("citations", []):
            chunk = chunks_by_id.get(cid)
            if chunk:
                print(f"  [{cid}] {chunk['text'][:600]}")
        print(f"\nJudge (claude-haiku-4-5) said: grounded={r['grounded']}  reasoning: {r['judge_reasoning']}")

        while True:
            answer = input("\nYour verdict (y/n/s): ").strip().lower()
            if answer in ("y", "n", "s"):
                break
            print("Please enter y, n, or s.")

        if answer != "s":
            results.append(
                {
                    "stage": r["stage"],
                    "question_id": r["question_id"],
                    "judge_grounded": r["grounded"],
                    "human_grounded": answer == "y",
                }
            )
        print()

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} human verdicts to {OUTPUT_PATH}")
    print("Run with --summarize to compute agreement.")


def summarize() -> None:
    if not OUTPUT_PATH.exists():
        print(f"No spot-check results yet at {OUTPUT_PATH} — run without --summarize first.")
        return

    results = json.loads(OUTPUT_PATH.read_text())
    if not results:
        print("No recorded verdicts.")
        return

    n = len(results)
    agree = sum(1 for r in results if r["judge_grounded"] == r["human_grounded"])
    agreement_rate = agree / n

    # Cohen's kappa: agreement corrected for the rate of agreement expected by chance.
    p_judge_true = sum(1 for r in results if r["judge_grounded"]) / n
    p_human_true = sum(1 for r in results if r["human_grounded"]) / n
    p_chance = p_judge_true * p_human_true + (1 - p_judge_true) * (1 - p_human_true)
    kappa = (agreement_rate - p_chance) / (1 - p_chance) if p_chance < 1 else float("nan")

    disagreements = [r for r in results if r["judge_grounded"] != r["human_grounded"]]

    print(f"n={n}  agreement={agreement_rate:.1%}  Cohen's kappa={kappa:.3f}")
    if disagreements:
        print("\nDisagreements:")
        for r in disagreements:
            print(f"  [{r['stage']} / {r['question_id']}] judge={r['judge_grounded']} human={r['human_grounded']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        summarize()
    else:
        run_review()


if __name__ == "__main__":
    main()
