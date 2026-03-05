#!/usr/bin/env python3
"""
採点結果の集計・モデル間比較スクリプト。

使い方:
    python scripts/aggregate_results.py --model_name claude
    python scripts/aggregate_results.py --all
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

CRITERIA = ["正確性", "文化的深度", "要点網羅性", "簡潔性"]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "type_a")


def load_scores(model_name):
    path = os.path.join(RESULTS_DIR, f"scores_{model_name}.jsonl")
    scores = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(json.loads(line))
    return scores


def aggregate_single_model(model_name):
    scores = load_scores(model_name)

    overall_by_criterion = defaultdict(list)
    overall_totals = []
    score_dist = {c: defaultdict(int) for c in CRITERIA}

    theme_by_criterion = defaultdict(lambda: defaultdict(list))
    theme_totals = defaultdict(list)

    for s in scores:
        total = s["total_score"]
        overall_totals.append(total)
        theme_totals[s["theme_name"]].append(total)

        for c in CRITERIA:
            val = s["scores"][c]["score"]
            overall_by_criterion[c].append(val)
            theme_by_criterion[s["theme_name"]][c].append(val)
            score_dist[c][str(val)] += 1

    summary = {
        "model_name": model_name,
        "judge_model": scores[0]["judge_model"] if scores else "unknown",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(scores),
        "overall": {
            "average_total": round(sum(overall_totals) / len(overall_totals), 2),
            "average_per_criterion": {
                c: round(sum(vals) / len(vals), 2)
                for c, vals in overall_by_criterion.items()
            },
            "score_distribution": {
                c: {str(i): dist.get(str(i), 0) for i in range(1, 6)}
                for c, dist in score_dist.items()
            },
        },
        "by_theme": {},
    }

    for theme in sorted(theme_totals.keys()):
        totals = theme_totals[theme]
        summary["by_theme"][theme] = {
            "count": len(totals),
            "average_total": round(sum(totals) / len(totals), 2),
            "average_per_criterion": {
                c: round(
                    sum(theme_by_criterion[theme][c]) / len(theme_by_criterion[theme][c]), 2
                )
                for c in CRITERIA
            },
        }

    output_path = os.path.join(RESULTS_DIR, f"summary_{model_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def create_comparison(model_names):
    summaries = {}
    for name in model_names:
        path = os.path.join(RESULTS_DIR, f"summary_{name}.json")
        with open(path, "r", encoding="utf-8") as f:
            summaries[name] = json.load(f)

    # 総合ランキング
    ranking = sorted(
        [
            {"model": name, "average_total": s["overall"]["average_total"]}
            for name, s in summaries.items()
        ],
        key=lambda x: x["average_total"],
        reverse=True,
    )

    # 観点別ランキング
    by_criterion = {}
    for c in CRITERIA:
        by_criterion[c] = sorted(
            [
                {"model": name, "average": s["overall"]["average_per_criterion"][c]}
                for name, s in summaries.items()
            ],
            key=lambda x: x["average"],
            reverse=True,
        )

    # テーマ別比較
    all_themes = set()
    for s in summaries.values():
        all_themes.update(s["by_theme"].keys())

    by_theme = {}
    for theme in sorted(all_themes):
        by_theme[theme] = {
            name: s["by_theme"].get(theme, {}).get("average_total")
            for name, s in summaries.items()
        }

    comparison = {
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "models": model_names,
        "overall_ranking": ranking,
        "by_criterion": by_criterion,
        "by_theme": by_theme,
    }

    output_path = os.path.join(RESULTS_DIR, "comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # ターミナルに結果表示
    print("\n" + "=" * 60)
    print("  総合ランキング")
    print("=" * 60)
    for i, r in enumerate(ranking, 1):
        print(f"  {i}. {r['model']:20s}  {r['average_total']:.2f} / 20.0")

    print("\n" + "=" * 60)
    print("  観点別スコア")
    print("=" * 60)
    header = f"  {'観点':<12s}"
    for name in model_names:
        header += f"  {name:>14s}"
    print(header)
    print("  " + "-" * (12 + 16 * len(model_names)))
    for c in CRITERIA:
        row = f"  {c:<12s}"
        for name in model_names:
            val = summaries[name]["overall"]["average_per_criterion"].get(c, 0)
            row += f"  {val:>14.2f}"
        print(row)

    print("\n" + "=" * 60)
    print("  テーマ別スコア")
    print("=" * 60)
    header = f"  {'テーマ':<8s}"
    for name in model_names:
        header += f"  {name:>14s}"
    print(header)
    print("  " + "-" * (8 + 16 * len(model_names)))
    for theme in sorted(all_themes):
        row = f"  {theme:<8s}"
        for name in model_names:
            val = by_theme[theme].get(name)
            row += f"  {val:>14.2f}" if val is not None else f"  {'N/A':>14s}"
        print(row)

    print()
    return comparison


def main():
    parser = argparse.ArgumentParser(description="採点結果の集計")
    parser.add_argument("--model_name", help="集計するモデル名")
    parser.add_argument("--all", action="store_true", help="全モデルを集計・比較")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.model_name:
        summary = aggregate_single_model(args.model_name)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.all:
        model_names = []
        for f in sorted(os.listdir(RESULTS_DIR)):
            if f.startswith("scores_") and f.endswith(".jsonl"):
                name = f[len("scores_") : -len(".jsonl")]
                aggregate_single_model(name)
                model_names.append(name)

        if not model_names:
            print("採点結果が見つかりません。")
            return

        if len(model_names) >= 2:
            create_comparison(model_names)
        else:
            print(f"モデル1つのみ ({model_names[0]})。比較には2つ以上必要です。")


if __name__ == "__main__":
    main()
