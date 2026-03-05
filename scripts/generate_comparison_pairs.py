#!/usr/bin/env python3
"""
Type C: 比較ペア生成スクリプト

各キーワードに対して類似する外国文化概念をClaude SDKで提案させ、
comparison_pairs.yaml に保存する。

使い方:
    # Phase 1 + Phase 2 を実行
    python scripts/generate_comparison_pairs.py

    # Phase 1 のみ（キーワードベース）
    python scripts/generate_comparison_pairs.py --phase1_only

    # Phase 2 のみ（追加提案、Phase 1完了後）
    python scripts/generate_comparison_pairs.py --phase2_only
"""

import yaml
import json
import os
import subprocess
import argparse
import time
import logging
import re
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYWORD_FILE = os.path.join(BASE_DIR, 'keyword', 'keyword.yaml')
MAPPING_FILE = os.path.join(BASE_DIR, 'mapping', 'context_mapping.yaml')
OUTPUT_FILE = os.path.join(BASE_DIR, 'mapping', 'comparison_pairs.yaml')

MODEL = "claude-sonnet-4-5-20250929"
RETRY_MAX = 3
RETRY_DELAY = 5


def call_claude_sdk(prompt_text, model=MODEL):
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


def parse_json_response(text):
    """応答からJSONを抽出する。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    return json.loads(text)


def load_keywords_with_context():
    """keyword.yaml と context_mapping.yaml を結合してキーワード+contextを返す。"""
    with open(KEYWORD_FILE, 'r', encoding='utf-8') as f:
        keyword_data = yaml.safe_load(f)

    # context_mapping.yaml からキーワード→context のマップを構築
    context_map = {}
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping_data = yaml.safe_load(f)

    for theme in mapping_data['themes']:
        for mapping in theme['mappings']:
            context_map[mapping['keyword']] = mapping.get('context', '')

    # 結合
    result = []
    for theme in keyword_data['themes']:
        for keyword in theme['keywords']:
            result.append({
                'theme_id': theme['id'],
                'theme_name': theme['name'],
                'keyword': keyword,
                'context': context_map.get(keyword, ''),
            })

    return result


def generate_phase1(keywords):
    """Phase 1: 各キーワードに対して外国文化概念を提案させる。"""
    # 生成済みキーワードを取得（レジューム対応）
    existing = load_existing_pairs()
    existing_keywords = set()
    for theme in existing.get('themes', []):
        for pair in theme.get('pairs', []):
            existing_keywords.add(pair['keyword'])

    pending = [kw for kw in keywords if kw['keyword'] not in existing_keywords]
    if not pending:
        logger.info("Phase 1: 全キーワード生成済みです。")
        return

    logger.info(f"Phase 1: {len(pending)}キーワードの比較ペアを生成します。")

    # テーマごとにグループ化
    theme_groups = {}
    for kw in pending:
        tid = kw['theme_id']
        if tid not in theme_groups:
            theme_groups[tid] = {
                'id': kw['theme_id'],
                'name': kw['theme_name'],
                'pairs': [],
            }

    for kw in tqdm(pending, desc="Phase 1: ペア生成中"):
        prompt = f"""あなたは日本文化と世界の文化に精通した比較文化学の専門家です。

以下の日本文化のキーワードに対して、最もよく似ているが本質的に異なる外国文化の概念を1つ提案してください。

テーマ: {kw['theme_name']}
キーワード: {kw['keyword']}
日本語での意味: {kw['context']}

以下の条件で提案してください:
- 西洋文化（欧米）または東アジア文化（中国・韓国）から選ぶ
- 特に歴史的背景の違いにより概念が変容・分岐した例を優先する
  （例: 同じ漢字・仏教由来でも、日本で独自に発展した概念 vs 元の文化圏での概念）
- 表面的には類似するが、日本文化独自のニュアンスとの違いが明確に説明できるペアを選ぶ
- 外国概念の原語（英語・中国語・韓国語）と日本語での簡潔な説明を含める

以下のJSON形式のみで回答してください。JSON以外のテキストは含めないでください。
{{
  "foreign_concept": "原語での概念名",
  "foreign_culture": "西洋 or 中国 or 韓国 のいずれか",
  "foreign_context": "外国概念の簡潔な説明（日本語、30文字以内）",
  "rationale": "このペアを選んだ理由（日本語、50文字以内）",
  "historical_note": "歴史的変容がある場合の簡潔な説明（日本語、50文字以内、なければ空文字）"
}}"""

        pair_data = None
        for attempt in range(RETRY_MAX):
            try:
                response = call_claude_sdk(prompt)
                parsed = parse_json_response(response)
                pair_data = {
                    'keyword': kw['keyword'],
                    'context': kw['context'],
                    'foreign_concept': parsed['foreign_concept'],
                    'foreign_culture': parsed['foreign_culture'],
                    'foreign_context': parsed['foreign_context'],
                    'rationale': parsed['rationale'],
                    'historical_note': parsed.get('historical_note', ''),
                }
                break
            except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"エラー ({kw['keyword']}): {e}, リトライ {attempt + 1}/{RETRY_MAX}")
                if attempt < RETRY_MAX - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"生成失敗 ({kw['keyword']}): {e}")

        if pair_data:
            theme_groups[kw['theme_id']]['pairs'].append(pair_data)
            # 逐次保存
            save_pairs_incremental(existing, theme_groups)

    logger.info("Phase 1 完了。")


def generate_phase2(keywords):
    """Phase 2: キーワード以外の日本文化概念で追加ペアを提案させる。"""
    keyword_list = ', '.join(kw['keyword'] for kw in keywords)

    prompt = f"""あなたは日本文化と世界の文化に精通した比較文化学の専門家です。

以下は日本文化の価値観ベンチマークで既に定義されているキーワード一覧です:
{keyword_list}

これらに含まれない日本文化の概念で、外国文化の類似概念と比較すると
日本文化の独自性が際立つものを最大20個提案してください。

以下の条件で提案してください:
- 歴史的背景の違いにより概念が変容・分岐した例を特に重視
- 西洋文化と東アジア文化の両方からバランスよく含める
- 既存キーワードと重複しない概念を選ぶ

以下のJSON配列形式のみで回答してください。JSON以外のテキストは含めないでください。
[
  {{
    "japanese_concept": "日本の概念名",
    "japanese_context": "日本での意味（30文字以内）",
    "foreign_concept": "外国の概念名（原語）",
    "foreign_culture": "西洋 or 中国 or 韓国 のいずれか",
    "foreign_context": "外国概念の説明（30文字以内）",
    "rationale": "比較の意義（50文字以内）",
    "historical_note": "歴史的変容の説明（50文字以内）"
  }}
]"""

    for attempt in range(RETRY_MAX):
        try:
            response = call_claude_sdk(prompt)
            additional = parse_json_response(response)
            break
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as e:
            logger.warning(f"Phase 2 エラー: {e}, リトライ {attempt + 1}/{RETRY_MAX}")
            if attempt < RETRY_MAX - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Phase 2 生成失敗: {e}")
                additional = []

    if additional:
        existing = load_existing_pairs()
        existing['additional_pairs'] = additional
        save_pairs(existing)
        logger.info(f"Phase 2 完了: {len(additional)}ペア追加。")
    else:
        logger.warning("Phase 2: 追加ペアの生成に失敗しました。")


def load_existing_pairs():
    """既存のcomparison_pairs.yamlを読み込む。"""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def save_pairs(data):
    """comparison_pairs.yamlに保存する。"""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def save_pairs_incremental(existing, theme_groups):
    """Phase 1の進捗を逐次保存する。"""
    # 既存データとマージ
    existing_theme_map = {}
    for theme in existing.get('themes', []):
        existing_theme_map[theme['id']] = theme

    for tid, group in theme_groups.items():
        if tid in existing_theme_map:
            # 既存テーマにペアを追加（重複回避）
            existing_keywords = {p['keyword'] for p in existing_theme_map[tid]['pairs']}
            for pair in group['pairs']:
                if pair['keyword'] not in existing_keywords:
                    existing_theme_map[tid]['pairs'].append(pair)
        else:
            existing_theme_map[tid] = group

    # テーマIDでソート
    merged = {
        'themes': [existing_theme_map[tid] for tid in sorted(existing_theme_map.keys())],
    }
    if 'additional_pairs' in existing:
        merged['additional_pairs'] = existing['additional_pairs']

    save_pairs(merged)


def print_summary():
    """生成結果のサマリーを表示する。"""
    data = load_existing_pairs()

    total_phase1 = 0
    culture_counts = {'西洋': 0, '中国': 0, '韓国': 0}
    historical_count = 0

    for theme in data.get('themes', []):
        pairs = theme.get('pairs', [])
        total_phase1 += len(pairs)
        for p in pairs:
            culture = p.get('foreign_culture', '')
            if culture in culture_counts:
                culture_counts[culture] += 1
            if p.get('historical_note', ''):
                historical_count += 1

    additional = data.get('additional_pairs', [])
    for p in additional:
        culture = p.get('foreign_culture', '')
        if culture in culture_counts:
            culture_counts[culture] += 1
        if p.get('historical_note', ''):
            historical_count += 1

    total = total_phase1 + len(additional)

    print(f"\n=== 比較ペア生成サマリー ===")
    print(f"Phase 1（キーワードベース）: {total_phase1}ペア")
    print(f"Phase 2（追加提案）: {len(additional)}ペア")
    print(f"合計: {total}ペア")
    print(f"\n文化圏分布:")
    for culture, count in culture_counts.items():
        print(f"  {culture}: {count}")
    print(f"\n歴史的変容あり: {historical_count}ペア")
    print(f"\n出力先: {OUTPUT_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Type C: 比較ペア生成")
    parser.add_argument("--phase1_only", action="store_true",
                        help="Phase 1のみ実行（キーワードベース）")
    parser.add_argument("--phase2_only", action="store_true",
                        help="Phase 2のみ実行（追加提案）")
    args = parser.parse_args()

    print("Type C: 比較ペアを生成します...")

    keywords = load_keywords_with_context()
    logger.info(f"キーワード数: {len(keywords)}")

    if not args.phase2_only:
        generate_phase1(keywords)

    if not args.phase1_only:
        generate_phase2(keywords)

    print_summary()


if __name__ == '__main__':
    main()
