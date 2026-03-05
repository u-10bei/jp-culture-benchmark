#!/usr/bin/env python3
"""
Type C: 類似概念比較問題の生成スクリプト

comparison_pairs.yaml から比較ペアを読み込み、
日本文化の概念と外国文化の類似概念を比較する問題を生成します。
Claude SDKで模範解答の下書きも生成します。

使い方:
    # 問題生成 + 模範解答下書き生成
    python scripts/generate_type_c.py

    # 問題生成のみ（模範解答下書き生成スキップ）
    python scripts/generate_type_c.py --questions_only
"""

import yaml
import json
import os
import subprocess
import argparse
import time
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS_FILE = os.path.join(BASE_DIR, 'mapping', 'comparison_pairs.yaml')
QUESTIONS_DIR = os.path.join(BASE_DIR, 'questions', 'type_c')
QUESTIONS_FILE = os.path.join(QUESTIONS_DIR, 'questions.jsonl')
ANSWERS_DIR = os.path.join(BASE_DIR, 'answers', 'type_c')
DRAFT_FILE = os.path.join(ANSWERS_DIR, 'answers_human_draft.jsonl')

DRAFT_MODEL = "claude-sonnet-4-5-20250929"
RETRY_MAX = 3
RETRY_DELAY = 5


def load_pairs():
    """comparison_pairs.yamlを読み込む"""
    with open(PAIRS_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def collect_all_pairs(data):
    """全テーマからペアを収集する"""
    pairs = []
    for theme in data['themes']:
        for pair in theme['pairs']:
            pairs.append({
                'theme_id': theme['id'],
                'theme_name': theme['name'],
                'keyword': pair['keyword'],
                'context': pair['context'],
                'foreign_concept': pair['foreign_concept'],
                'foreign_culture': pair['foreign_culture'],
                'foreign_context': pair['foreign_context'],
                'rationale': pair.get('rationale', ''),
                'historical_note': pair.get('historical_note', ''),
            })
    return pairs


def generate_questions(pairs):
    """問題を生成する"""
    questions = []

    for i, pair in enumerate(pairs, 1):
        theme_name = pair['theme_name']
        keyword = pair['keyword']
        context = pair['context']
        foreign_concept = pair['foreign_concept']
        foreign_culture = pair['foreign_culture']
        foreign_context = pair['foreign_context']

        # プロンプト形式の決定
        if theme_name == keyword:
            prompt = (
                f"日本文化の「{keyword}」（{context}）と、"
                f"{foreign_culture}文化の「{foreign_concept}」（{foreign_context}）は"
                f"似た概念ですが、両者の違いを日本文化の特徴を踏まえて説明してください。"
            )
        else:
            prompt = (
                f"日本文化の「{theme_name}」に関連する「{keyword}」（{context}）と、"
                f"{foreign_culture}文化の「{foreign_concept}」（{foreign_context}）は"
                f"似た概念ですが、両者の違いを日本文化の特徴を踏まえて説明してください。"
            )

        question = {
            "id": f"C-{i:03d}",
            "type": "C",
            "theme_id": pair['theme_id'],
            "theme_name": theme_name,
            "keyword": keyword,
            "context": context,
            "foreign_concept": foreign_concept,
            "foreign_culture": foreign_culture,
            "foreign_context": foreign_context,
            "historical_note": pair.get('historical_note', ''),
            "prompt": prompt,
            "constraints": {
                "language": "ja",
                "max_chars": 200
            }
        }
        questions.append(question)

    return questions


def save_questions(questions):
    """JSONL形式で保存する"""
    os.makedirs(QUESTIONS_DIR, exist_ok=True)

    with open(QUESTIONS_FILE, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')


def call_claude_sdk(prompt_text, model=DRAFT_MODEL):
    """Claude Code SDK (claude CLI) を使ってプロンプトを送信し応答を得る。"""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    result = subprocess.run(
        ["claude", "-p", prompt_text, "--model", model, "--output-format", "text"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI エラー: {result.stderr.strip()}")
    return result.stdout.strip()


def generate_draft_answers(questions):
    """Claude SDKで模範解答の下書きを生成する"""
    os.makedirs(ANSWERS_DIR, exist_ok=True)

    # 生成済みIDを取得（レジューム対応）
    existing_ids = set()
    if os.path.exists(DRAFT_FILE):
        with open(DRAFT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)['id'])

    pending = [q for q in questions if q['id'] not in existing_ids]
    if not pending:
        logger.info("全問の下書き生成済みです。")
        return

    logger.info(f"模範解答下書き生成: {len(pending)}問")

    with open(DRAFT_FILE, 'a', encoding='utf-8') as out:
        for q in tqdm(pending, desc="下書き生成中"):
            draft_prompt = f"""あなたは日本文化と比較文化学の専門家です。以下の質問に対する模範解答を作成してください。

テーマ: {q['theme_name']}
日本の概念: {q['keyword']}（{q['context']}）
外国の概念: {q['foreign_concept']}（{q['foreign_context']}、{q['foreign_culture']}文化）

質問: {q['prompt']}

以下の条件で模範解答を作成してください:
- 200文字以内
- 両概念の共通点に触れつつ、日本文化の概念に固有のニュアンスを明確にする
- 文化的な深みと正確性を重視
- 模範解答のみを出力（説明や注釈は不要）"""

            for attempt in range(RETRY_MAX):
                try:
                    answer_text = call_claude_sdk(draft_prompt)
                    break
                except (subprocess.TimeoutExpired, RuntimeError) as e:
                    logger.warning(f"エラー ({q['id']}): {e}, リトライ {attempt + 1}/{RETRY_MAX}")
                    if attempt < RETRY_MAX - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.error(f"下書き生成失敗 ({q['id']}): {e}")
                        answer_text = ""

            record = {
                "id": q['id'],
                "theme_name": q['theme_name'],
                "keyword": q['keyword'],
                "foreign_concept": q['foreign_concept'],
                "foreign_culture": q['foreign_culture'],
                "prompt": q['prompt'],
                "answer": answer_text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush()

    logger.info(f"下書き生成完了: {DRAFT_FILE}")


def print_summary(questions):
    """生成結果のサマリーを表示する"""
    culture_counts = {}
    theme_counts = {}
    for q in questions:
        culture = q['foreign_culture']
        culture_counts[culture] = culture_counts.get(culture, 0) + 1
        theme = q['theme_name']
        theme_counts[theme] = theme_counts.get(theme, 0) + 1

    print(f"\n=== Type C 問題生成サマリー ===")
    print(f"総問題数: {len(questions)}")
    print(f"\n文化圏分布:")
    for culture, count in sorted(culture_counts.items(), key=lambda x: -x[1]):
        print(f"  {culture}: {count}")
    print(f"\nテーマ別問題数:")
    for theme, count in sorted(theme_counts.items(), key=lambda x: -x[1]):
        print(f"  {theme}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Type C: 類似概念比較問題の生成")
    parser.add_argument("--questions_only", action="store_true",
                        help="問題生成のみ（模範解答下書き生成スキップ）")
    args = parser.parse_args()

    print("Type C: 類似概念比較問題を生成します...")

    # ペア読み込み
    data = load_pairs()
    pairs = collect_all_pairs(data)
    logger.info(f"比較ペア数: {len(pairs)}")

    # 問題生成
    questions = generate_questions(pairs)

    # 保存
    save_questions(questions)

    print(f"問題生成完了: {len(questions)}問")
    print(f"出力先: {QUESTIONS_FILE}")

    # サンプル表示
    print("\n=== サンプル（最初の3問）===")
    for q in questions[:3]:
        print(json.dumps(q, ensure_ascii=False, indent=2))
        print()

    # サマリー
    print_summary(questions)

    # 模範解答下書き生成
    if not args.questions_only:
        generate_draft_answers(questions)
    else:
        print("--questions_only が指定されたため、模範解答下書き生成をスキップします。")


if __name__ == '__main__':
    main()
