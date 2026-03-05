#!/usr/bin/env python3
"""
Type B: 場面適用問題の生成スクリプト

キーワードが具体的な文化要素においてどう表れるかを問う問題を生成します。
context_mapping.yaml から100問を選定し、Claude SDKで模範解答の下書きを生成します。

使い方:
    # 問題生成 + 模範解答下書き生成
    python scripts/generate_type_b.py

    # 問題生成のみ（模範解答下書き生成スキップ）
    python scripts/generate_type_b.py --questions_only
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
MAPPING_FILE = os.path.join(BASE_DIR, 'mapping', 'context_mapping.yaml')
QUESTIONS_DIR = os.path.join(BASE_DIR, 'questions', 'type_b')
QUESTIONS_FILE = os.path.join(QUESTIONS_DIR, 'questions.jsonl')
ANSWERS_DIR = os.path.join(BASE_DIR, 'answers', 'type_b')
DRAFT_FILE = os.path.join(ANSWERS_DIR, 'answers_human_draft.jsonl')

DRAFT_MODEL = "claude-sonnet-4-5-20250929"
RETRY_MAX = 3
RETRY_DELAY = 5


def load_mapping():
    """context_mapping.yamlを読み込む"""
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def select_pairs(data):
    """100ペアを選定する

    1. 各キーワードから文化要素を1つ選択（cultural_elementsが空のものは除外）
       - 同語反復回避: 先頭が抽象的（仏教/神道）なら、2番目に具体的な要素があればそちらを採用
    2. 茶道の過度な集中を回避: 代替要素がある場合は積極的に変更
    3. 残り1問は未使用要素の多いキーワードから追加して合計100問に
    """
    from collections import Counter

    # 仏教/神道は広すぎて同語反復になりやすい抽象カテゴリ
    ABSTRACT_ELEMENTS = {"仏教", "神道"}
    # 集中しやすい要素の上限（これを超えたら代替を選ぶ）
    CONCENTRATION_LIMIT = 4

    # --- Pass 1: 初期選択（同語反復回避のみ） ---
    candidates = []  # (pair_dict, cultural_elements_list) のタプル

    for theme in data['themes']:
        theme_id = theme['id']
        theme_name = theme['name']

        for mapping in theme['mappings']:
            keyword = mapping['keyword']
            cultural_elements = mapping.get('cultural_elements', [])
            context = mapping.get('context', '')

            if not cultural_elements:
                logger.info(f"除外（文化要素なし）: {keyword}")
                continue

            # 文化要素の選択: 同語反復回避
            selected = cultural_elements[0]
            if selected in ABSTRACT_ELEMENTS and len(cultural_elements) >= 2:
                if cultural_elements[1] not in ABSTRACT_ELEMENTS:
                    selected = cultural_elements[1]
                    logger.info(f"同語反復回避: {keyword} → {cultural_elements[0]} を {selected} に変更")

            pair = {
                'theme_id': theme_id,
                'theme_name': theme_name,
                'keyword': keyword,
                'cultural_element': selected,
                'context': context,
            }
            candidates.append((pair, cultural_elements))

    # --- Pass 2: 集中回避（茶道など） ---
    element_counts = Counter(p['cultural_element'] for p, _ in candidates)
    over_concentrated = {e for e, c in element_counts.items() if c > CONCENTRATION_LIMIT}

    if over_concentrated:
        logger.info(f"集中回避対象: {over_concentrated}")

    for i, (pair, cultural_elements) in enumerate(candidates):
        selected = pair['cultural_element']
        if selected not in over_concentrated:
            continue
        if element_counts[selected] <= CONCENTRATION_LIMIT:
            continue

        # 代替候補: 抽象要素と集中要素を除く
        alternatives = [
            e for e in cultural_elements
            if e != selected and e not in ABSTRACT_ELEMENTS
        ]
        if not alternatives:
            continue

        # 代替候補のうち最もカウントが少ないものを選ぶ
        best_alt = min(alternatives, key=lambda e: element_counts.get(e, 0))
        logger.info(
            f"集中回避: {pair['keyword']} → {selected}（{element_counts[selected]}回）"
            f"を {best_alt}（{element_counts.get(best_alt, 0)}回）に変更"
        )
        element_counts[selected] -= 1
        element_counts[best_alt] = element_counts.get(best_alt, 0) + 1
        candidates[i] = (
            {**pair, 'cultural_element': best_alt},
            cultural_elements,
        )

    # --- 結果構築 ---
    pairs = [pair for pair, _ in candidates]

    # 100問目候補
    best_extra = None
    best_extra_remaining = 0
    for pair, cultural_elements in candidates:
        remaining = [e for e in cultural_elements if e != pair['cultural_element']]
        if len(remaining) > best_extra_remaining:
            best_extra_remaining = len(remaining)
            best_extra = {
                'theme_id': pair['theme_id'],
                'theme_name': pair['theme_name'],
                'keyword': pair['keyword'],
                'cultural_element': remaining[0],
                'context': pair['context'],
            }

    logger.info(f"選定: {len(pairs)}ペア")

    # 100問に足りない場合、未使用要素の最も多いキーワードから追加
    if len(pairs) < 100 and best_extra:
        pairs.append(best_extra)
        logger.info(
            f"追加: {best_extra['keyword']} → {best_extra['cultural_element']}"
            f"（残り要素数: {best_extra_remaining}）"
        )

    # 最終的な要素分布をログ出力
    final_counts = Counter(p['cultural_element'] for p in pairs)
    top10 = final_counts.most_common(10)
    logger.info(f"文化要素 上位10: {top10}")

    return pairs


def generate_questions(pairs):
    """問題を生成する"""
    questions = []

    for i, pair in enumerate(pairs, 1):
        theme_name = pair['theme_name']
        keyword = pair['keyword']
        cultural_element = pair['cultural_element']
        context = pair['context']

        # プロンプト形式の決定（contextを括弧書きで補足）
        if theme_name == keyword:
            prompt = (
                f"日本文化の「{theme_name}」という概念（{context}）は、"
                f"{cultural_element}においてどのように表れていますか？"
                f"具体的に説明してください。"
            )
        else:
            prompt = (
                f"日本文化の「{theme_name}」に関連する「{keyword}」（{context}）は、"
                f"{cultural_element}においてどのように表れていますか？"
                f"具体的に説明してください。"
            )

        question = {
            "id": f"B-{i:03d}",
            "type": "B",
            "theme_id": pair['theme_id'],
            "theme_name": theme_name,
            "keyword": keyword,
            "cultural_element": cultural_element,
            "context": context,
            "prompt": prompt,
            "constraints": {
                "language": "ja",
                "max_chars": 150
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
            draft_prompt = f"""あなたは日本文化の専門家です。以下の質問に対する模範解答を作成してください。

テーマ: {q['theme_name']}
キーワード: {q['keyword']}
文化要素: {q['cultural_element']}
文脈ヒント: {q['context']}

質問: {q['prompt']}

以下の条件で模範解答を作成してください:
- 150文字以内
- キーワードが文化要素においてどのように具体的に表れているかを説明
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
                "cultural_element": q['cultural_element'],
                "prompt": q['prompt'],
                "answer": answer_text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush()

    logger.info(f"下書き生成完了: {DRAFT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Type B: 場面適用問題の生成")
    parser.add_argument("--questions_only", action="store_true",
                        help="問題生成のみ（模範解答下書き生成スキップ）")
    args = parser.parse_args()

    print("Type B: 場面適用問題を生成します...")

    # マッピング読み込み
    data = load_mapping()

    # ペア選定
    pairs = select_pairs(data)

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

    # 模範解答下書き生成
    if not args.questions_only:
        generate_draft_answers(questions)
    else:
        print("--questions_only が指定されたため、模範解答下書き生成をスキップします。")


if __name__ == '__main__':
    main()
