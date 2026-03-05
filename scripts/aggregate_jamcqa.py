#!/usr/bin/env python3
"""
JamC-QA 集計・モデル間比較スクリプト。

使い方:
    python scripts/aggregate_jamcqa.py --model_name qwen3-8b
    python scripts/aggregate_jamcqa.py --all
"""

import argparse
import json
import os
import glob
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "jamcqa"


def aggregate_single(model_name):
    answers_path = RESULTS_DIR / f"answers_{model_name}.jsonl"
    if not answers_path.exists():
        print(f"  結果ファイルが見つかりません: {answers_path}")
        return None

    with open(answers_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])

    # カテゴリ別
    cat_stats = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"correct": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        if r["is_correct"]:
            cat_stats[cat]["correct"] += 1

    cat_accuracy = {}
    for cat, stats in cat_stats.items():
        cat_accuracy[cat] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": round(stats["correct"] / stats["total"], 4) if stats["total"] > 0 else 0,
        }

    summary = {
        "model_name": model_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": total,
        "overall": {
            "correct": correct,
            "accuracy": round(correct / total, 4) if total > 0 else 0,
        },
        "by_category": cat_accuracy,
    }

    summary_path = RESULTS_DIR / f"summary_{model_name}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def print_comparison(summaries):
    models = sorted(summaries.keys(), key=lambda m: -summaries[m]["overall"]["accuracy"])

    print("\n" + "=" * 60)
    print("  JamC-QA 総合ランキング")
    print("=" * 60)
    for i, model in enumerate(models, 1):
        s = summaries[model]
        acc = s["overall"]["accuracy"] * 100
        correct = s["overall"]["correct"]
        total = s["total_questions"]
        print(f"  {i}. {model:<20s} {acc:5.1f}% ({correct}/{total})")

    # カテゴリ一覧
    all_cats = set()
    for s in summaries.values():
        all_cats.update(s["by_category"].keys())
    all_cats = sorted(all_cats)

    if all_cats:
        print("\n" + "=" * 60)
        print("  カテゴリ別 Accuracy (%)")
        print("=" * 60)
        header = f"  {'カテゴリ':<20s}" + "".join(f"{m:>14s}" for m in models)
        print(header)
        print("  " + "-" * (20 + 14 * len(models)))
        for cat in all_cats:
            row = f"  {cat:<20s}"
            for model in models:
                cat_data = summaries[model]["by_category"].get(cat)
                if cat_data:
                    acc = cat_data["accuracy"] * 100
                    row += f"{acc:>13.1f}%"
                else:
                    row += f"{'N/A':>14s}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description="JamC-QA集計・比較")
    parser.add_argument("--model_name", help="単一モデルの集計")
    parser.add_argument("--all", action="store_true", help="全モデルの集計・比較")
    args = parser.parse_args()

    if args.model_name:
        summary = aggregate_single(args.model_name)
        if summary:
            acc = summary["overall"]["accuracy"] * 100
            print(f"\n{args.model_name}: {acc:.1f}%")

    elif args.all:
        pattern = str(RESULTS_DIR / "answers_*.jsonl")
        files = glob.glob(pattern)
        if not files:
            print("結果ファイルが見つかりません")
            return

        summaries = {}
        for f in files:
            model = Path(f).stem.replace("answers_", "")
            summary = aggregate_single(model)
            if summary:
                summaries[model] = summary

        if summaries:
            # comparison.json 出力
            comparison = {
                "models": list(summaries.keys()),
                "overall_ranking": sorted(
                    [{"model": m, "accuracy": s["overall"]["accuracy"]}
                     for m, s in summaries.items()],
                    key=lambda x: -x["accuracy"],
                ),
                "by_category": {},
            }
            all_cats = set()
            for s in summaries.values():
                all_cats.update(s["by_category"].keys())
            for cat in sorted(all_cats):
                comparison["by_category"][cat] = {
                    m: s["by_category"].get(cat, {}).get("accuracy", 0)
                    for m, s in summaries.items()
                }
            comp_path = RESULTS_DIR / "comparison.json"
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)

            print_comparison(summaries)


if __name__ == "__main__":
    main()
