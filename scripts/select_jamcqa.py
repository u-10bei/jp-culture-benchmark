#!/usr/bin/env python3
"""
JamC-QAから本プロジェクトに関連度の高い100問を選定する。

文化・風習・風土の3カテゴリ(1,237問)からClaude Code SDKで関連度を評価し、
上位100問を選定して questions/jamcqa/test.jsonl に保存する。

使い方:
    python scripts/select_jamcqa.py
    python scripts/select_jamcqa.py --resume   # 途中再開
"""

import argparse
import json
import os
import re
import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_CATEGORIES = {"culture", "custom", "regional_identity"}
BATCH_SIZE = 10
SELECT_COUNT = 100
JUDGE_MODEL = "claude-sonnet-4-5-20250929"

THEMES = "和・型・道・気・節・情・忠・神・仏・縁・信・徳・美"

SELECTION_PROMPT_TEMPLATE = """あなたは日本文化の専門家です。

本プロジェクトは、日本人の精神的価値観を評価するベンチマークです。
13テーマ: {themes}

以下の多肢選択問題が、「日本の文化的価値観・精神性の理解」にどの程度関連するかを1-5で評価してください。

## 評価基準
5: 直接的に価値観・精神性を問う（例: 礼儀作法の意味、和の精神に関する問い）
4: 価値観が背景にある文化・風習を問う（例: 茶道の作法、武道の精神）
3: 日本文化の知識を問うが価値観との関連は間接的（例: 祭りの名称、伝統芸能の事実）
2: 事実知識が中心で価値観との関連が薄い（例: 地名の由来、特産品）
1: 価値観とほぼ無関係（例: 統計データ、現代の制度）

## 問題一覧
{questions_text}

## 回答形式
以下のJSON配列のみを出力してください。説明文は不要です。
[{{"qid": "問題ID", "score": 評価点, "reason": "10文字程度の簡潔な理由"}}]"""


def call_claude_sdk(prompt_text, model=JUDGE_MODEL):
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    result = subprocess.run(
        ["claude", "-p", prompt_text, "--model", model, "--output-format", "text"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI エラー: {result.stderr.strip()}")
    return result.stdout.strip()


def parse_selection_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    return json.loads(text)


def format_question(q, idx):
    choices = f"A. {q['choice0']}  B. {q['choice1']}  C. {q['choice2']}  D. {q['choice3']}"
    return f"[{idx}] qid={q['qid']} カテゴリ={q['category']}\n問題: {q['question']}\n{choices}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="途中再開")
    args = parser.parse_args()

    test_all_path = BASE_DIR / "questions" / "jamcqa" / "test_all.jsonl"
    log_path = BASE_DIR / "questions" / "jamcqa" / "selection_log.jsonl"
    output_path = BASE_DIR / "questions" / "jamcqa" / "test.jsonl"

    # 全問読み込み → 対象3カテゴリ抽出
    with open(test_all_path, "r", encoding="utf-8") as f:
        all_questions = [json.loads(line) for line in f if line.strip()]

    target_questions = [q for q in all_questions if q["category"] in TARGET_CATEGORIES]
    logger.info(f"対象カテゴリ: {TARGET_CATEGORIES}")
    logger.info(f"対象問題数: {len(target_questions)}問 (全{len(all_questions)}問中)")

    # resume: 既にスコア済みのqidを除外
    scored = {}
    if args.resume and log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    scored[entry["qid"]] = entry
        logger.info(f"既存スコア: {len(scored)}件")

    pending = [q for q in target_questions if q["qid"] not in scored]
    logger.info(f"未処理: {len(pending)}問")

    # バッチ処理
    batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    total_batches = len(batches)

    for batch_idx, batch in enumerate(batches):
        logger.info(f"バッチ {batch_idx + 1}/{total_batches} ({len(batch)}問)")

        questions_text = "\n\n".join(
            format_question(q, i + 1) for i, q in enumerate(batch)
        )
        prompt = SELECTION_PROMPT_TEMPLATE.format(
            themes=THEMES, questions_text=questions_text
        )

        for attempt in range(3):
            try:
                response = call_claude_sdk(prompt)
                results = parse_selection_response(response)

                with open(log_path, "a", encoding="utf-8") as f:
                    for item in results:
                        scored[item["qid"]] = item
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                break
            except (json.JSONDecodeError, subprocess.TimeoutExpired, RuntimeError) as e:
                logger.warning(f"バッチ {batch_idx + 1} 失敗 (attempt {attempt + 1}): {e}")
                time.sleep(5)
        else:
            logger.error(f"バッチ {batch_idx + 1} を3回失敗、スキップ")

    # 上位100問を選定
    logger.info(f"\nスコア済み: {len(scored)}問")

    all_scored = sorted(scored.values(), key=lambda x: -x.get("score", 0))
    selected_qids = {item["qid"] for item in all_scored[:SELECT_COUNT]}

    # スコア分布を表示
    score_dist = {}
    for item in scored.values():
        s = item.get("score", 0)
        score_dist[s] = score_dist.get(s, 0) + 1
    logger.info("スコア分布:")
    for s in sorted(score_dist.keys(), reverse=True):
        marker = " ← 選定ライン" if sum(score_dist.get(i, 0) for i in range(s, 6)) <= SELECT_COUNT else ""
        logger.info(f"  {s}点: {score_dist[s]}問{marker}")

    # 選定問題をtest.jsonlに保存
    selected_questions = [q for q in target_questions if q["qid"] in selected_qids]
    with open(output_path, "w", encoding="utf-8") as f:
        for q in selected_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info(f"\n選定完了: {len(selected_questions)}問 → {output_path}")

    # カテゴリ別内訳
    cat_counts = {}
    for q in selected_questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count}問")


if __name__ == "__main__":
    main()
