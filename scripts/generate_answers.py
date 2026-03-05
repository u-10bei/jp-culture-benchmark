#!/usr/bin/env python3
"""
vLLM OpenAI互換API経由で回答を生成するスクリプト。

使い方:
    # vLLMサーバー起動後に実行
    python scripts/generate_answers.py --model_name llmjp-v4-8b
    python scripts/generate_answers.py --model_name qwen3-8b
    python scripts/generate_answers.py --model_name qwen3-8b --api_base http://localhost:8001/v1
"""

import argparse
import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_FILE = os.path.join(BASE_DIR, "questions", "type_a", "questions_claude.jsonl")
ANSWERS_DIR = os.path.join(BASE_DIR, "answers", "type_a")


def load_questions(path):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_existing_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["id"])
    return ids


def generate_answer(client, model_id, question, max_chars):
    system_prompt = (
        "あなたは日本文化に精通した専門家です。"
        f"質問に日本語で簡潔に答えてください。回答は{max_chars}文字以内としてください。"
    )
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question["prompt"]},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="vLLM経由で回答を生成")
    parser.add_argument("--model_name", required=True, help="出力ファイル名に使うモデル名")
    parser.add_argument("--api_base", default="http://localhost:8000/v1", help="vLLM APIのベースURL")
    parser.add_argument("--model_id", default=None, help="vLLMサーバー上のモデルID（省略時は自動検出）")
    parser.add_argument("--questions_file", default=QUESTIONS_FILE)
    parser.add_argument("--output_dir", default=ANSWERS_DIR)
    args = parser.parse_args()

    client = OpenAI(base_url=args.api_base, api_key="dummy")

    # モデルIDの自動検出
    if args.model_id is None:
        models = client.models.list()
        args.model_id = models.data[0].id
        print(f"モデルID自動検出: {args.model_id}")

    questions = load_questions(args.questions_file)
    output_file = os.path.join(args.output_dir, f"answers_{args.model_name}.jsonl")
    existing_ids = load_existing_ids(output_file)

    if existing_ids:
        print(f"レジューム: {len(existing_ids)}件は生成済み")

    pending = [q for q in questions if q["id"] not in existing_ids]
    if not pending:
        print("全問生成済みです。")
        return

    print(f"生成対象: {len(pending)}問")

    with open(output_file, "a", encoding="utf-8") as out:
        for q in tqdm(pending, desc=f"生成中 ({args.model_name})"):
            max_chars = q["constraints"]["max_chars"]
            try:
                answer_text = generate_answer(client, args.model_id, q, max_chars)
            except Exception as e:
                print(f"\nエラー ({q['id']}): {e}")
                continue

            record = {
                "id": q["id"],
                "theme_name": q["theme_name"],
                "keyword": q["keyword"],
                "prompt": q["prompt"],
                "answer": answer_text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    print(f"完了: {output_file}")


if __name__ == "__main__":
    main()
