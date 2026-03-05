#!/usr/bin/env python3
"""
Type A: 意味説明問題の生成スクリプト

キーワードの意味を日本文化の文脈で説明させる問題を生成します。
"""

import yaml
import json
import os

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYWORD_FILE = os.path.join(BASE_DIR, 'keyword', 'keyword.yaml')
OUTPUT_DIR = os.path.join(BASE_DIR, 'questions', 'type_a')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'questions.jsonl')


def load_keywords():
    """keyword.yamlを読み込む"""
    with open(KEYWORD_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_questions(data):
    """問題を生成する"""
    questions = []
    question_id = 1

    for theme in data['themes']:
        theme_id = theme['id']
        theme_name = theme['name']

        for keyword in theme['keywords']:
            # テーマ名とキーワードが同じ場合はプロンプトと文字数を変更
            if theme_name == keyword:
                prompt = f"日本文化の「{theme_name}」という概念を説明してください。"
                max_chars = 200
            else:
                prompt = f"日本文化の「{theme_name}」という概念に関連して、「{keyword}」とは何か説明してください。"
                max_chars = 100

            question = {
                "id": f"A-{question_id:03d}",
                "type": "A",
                "theme_id": theme_id,
                "theme_name": theme_name,
                "keyword": keyword,
                "prompt": prompt,
                "constraints": {
                    "language": "ja",
                    "max_chars": max_chars
                }
            }
            questions.append(question)
            question_id += 1

    return questions


def save_questions(questions):
    """JSONL形式で保存する"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')


def main():
    print("Type A: 意味説明問題を生成します...")
    
    # キーワード読み込み
    data = load_keywords()
    
    # 問題生成
    questions = generate_questions(data)
    
    # 保存
    save_questions(questions)
    
    print(f"生成完了: {len(questions)}問")
    print(f"出力先: {OUTPUT_FILE}")
    
    # サンプル表示
    print("\n=== サンプル（最初の3問）===")
    for q in questions[:3]:
        print(json.dumps(q, ensure_ascii=False, indent=2))
        print()


if __name__ == '__main__':
    main()
