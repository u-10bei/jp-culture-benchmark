#!/usr/bin/env python3
"""
JamC-QA 多肢選択式ベンチマーク評価スクリプト。

vLLMオフライン推論（ローカルモデル）またはClaude Code SDKで回答を生成し、
Exact Match accuracyで評価する。

使い方:
    # ローカルモデル（run_vllm.sh経由）
    bash scripts/run_vllm.sh eval_jamcqa.py --model_name qwen3-8b --model_path Qwen/Qwen3-8B

    # Claude
    python scripts/eval_jamcqa.py --model_name claude

    # 途中再開
    python scripts/eval_jamcqa.py --model_name qwen3-8b --resume
"""

import argparse
import json
import os
import re
import subprocess
import time
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def strip_think_tags(text):
    """思考過程・推論トークンを除去して回答本文のみを返す。"""
    # 1. <think>...</think> 完全ペア
    result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # 2. 開始タグなしの </think>（テンプレートが <think> を注入するモデル）
    if "</think>" in result:
        result = re.sub(r"^.*?</think>\s*", "", result, flags=re.DOTALL)
    # 3. gpt-oss チャンネルマーカー（analysis...final の後が回答）
    if result.startswith("analysis"):
        m = re.search(r"(?:assistant)?final\s*", result)
        if m:
            result = result[m.end():]
    # 4. </think>なしの不完全な思考出力（空回答とする）
    if re.match(r"^(Okay|Let me|I need|The user|First,|So,|Alright|Hmm)", result):
        return ""
    return result.strip()


def load_few_shot_examples(dev_path):
    """devセットから4-shot用の例を読み込む（先頭4問）。"""
    examples = []
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
            if len(examples) >= 4:
                break
    return examples


def format_mc_question(q, include_answer=False):
    """多肢選択問題をテキストに変換。"""
    text = f"問題: {q['question']}\n"
    text += f"A. {q['choice0']}\nB. {q['choice1']}\nC. {q['choice2']}\nD. {q['choice3']}"
    if include_answer:
        correct_letter = INDEX_TO_LETTER[q["answer_index"]]
        text += f"\n正解: {correct_letter}"
    return text


def build_mc_prompt(few_shot_examples, test_question):
    """4-shot + テスト問題のプロンプトを構築。"""
    prompt = "以下は日本に関する4択問題です。正解の選択肢をA/B/C/Dの1文字で答えてください。\n\n"
    for ex in few_shot_examples:
        prompt += format_mc_question(ex, include_answer=True) + "\n\n"
    prompt += format_mc_question(test_question, include_answer=False) + "\n正解: "
    return prompt


def extract_answer_letter(text):
    """モデル出力からA/B/C/Dを抽出。"""
    text = text.strip()
    if text and text[0] in LETTER_TO_INDEX:
        return text[0]
    match = re.search(r"[ABCD]", text)
    if match:
        return match.group(0)
    return None


# ============================================================
# vLLMオフライン推論
# ============================================================

def evaluate_vllm(model_path, test_questions, few_shot_examples, output_path, resume_qids):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pending = [q for q in test_questions if q["qid"] not in resume_qids]
    logger.info(f"vLLM推論: {len(pending)}問 (モデル: {model_path})")

    if not pending:
        logger.info("全問処理済み")
        return

    # プロンプト構築
    prompts = []
    for q in pending:
        mc_prompt = build_mc_prompt(few_shot_examples, q)
        messages = [{"role": "user", "content": mc_prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    # vLLM推論
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    logger.info("vLLM推論開始...")
    outputs = llm.generate(prompts, sampling_params)

    # 結果保存
    with open(output_path, "a", encoding="utf-8") as f:
        for q, output in zip(pending, outputs):
            raw_text = strip_think_tags(output.outputs[0].text)
            predicted = extract_answer_letter(raw_text)
            correct = INDEX_TO_LETTER[q["answer_index"]]
            record = {
                "qid": q["qid"],
                "category": q["category"],
                "question": q["question"],
                "predicted": predicted,
                "correct": correct,
                "is_correct": predicted == correct,
                "raw_output": raw_text[:100],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"推論完了: {output_path}")

    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================
# Claude Code SDK
# ============================================================

def call_claude_sdk(prompt_text, model=JUDGE_MODEL):
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    result = subprocess.run(
        ["claude", "-p", prompt_text, "--model", model, "--output-format", "text"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI エラー: {result.stderr.strip()}")
    return result.stdout.strip()


def evaluate_claude(test_questions, few_shot_examples, output_path, resume_qids):
    pending = [q for q in test_questions if q["qid"] not in resume_qids]
    logger.info(f"Claude評価: {len(pending)}問")

    if not pending:
        logger.info("全問処理済み")
        return

    with open(output_path, "a", encoding="utf-8") as f:
        for q in tqdm(pending, desc="Claude評価中"):
            mc_prompt = build_mc_prompt(few_shot_examples, q)
            prompt = mc_prompt + "\n\n上記の問題の正解をA/B/C/Dの1文字のみで答えてください。"

            predicted = None
            for attempt in range(3):
                try:
                    response = call_claude_sdk(prompt)
                    predicted = extract_answer_letter(response)
                    if predicted:
                        break
                except (subprocess.TimeoutExpired, RuntimeError) as e:
                    logger.warning(f"リトライ {attempt + 1}: {e}")
                    time.sleep(5)

            correct = INDEX_TO_LETTER[q["answer_index"]]
            record = {
                "qid": q["qid"],
                "category": q["category"],
                "question": q["question"],
                "predicted": predicted,
                "correct": correct,
                "is_correct": predicted == correct,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"評価完了: {output_path}")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="JamC-QA多肢選択式ベンチマーク評価")
    parser.add_argument("--model_name", required=True, help="モデル名（出力ファイル名に使用）")
    parser.add_argument("--model_path", help="vLLMモデルパス（ローカルモデル用）")
    parser.add_argument("--resume", action="store_true", help="途中再開")
    args = parser.parse_args()

    test_path = BASE_DIR / "questions" / "jamcqa" / "test.jsonl"
    dev_path = BASE_DIR / "questions" / "jamcqa" / "dev.jsonl"
    output_path = BASE_DIR / "results" / "jamcqa" / f"answers_{args.model_name}.jsonl"
    os.makedirs(output_path.parent, exist_ok=True)

    # テスト問題読み込み
    with open(test_path, "r", encoding="utf-8") as f:
        test_questions = [json.loads(line) for line in f if line.strip()]
    logger.info(f"テスト問題: {len(test_questions)}問")

    # Few-shot例読み込み
    few_shot_examples = load_few_shot_examples(dev_path)
    logger.info(f"Few-shot例: {len(few_shot_examples)}問")

    # Resume: 既存結果の読み込み
    resume_qids = set()
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    resume_qids.add(json.loads(line)["qid"])
        logger.info(f"既存回答: {len(resume_qids)}件")

    # 評価実行
    if args.model_path:
        evaluate_vllm(args.model_path, test_questions, few_shot_examples, output_path, resume_qids)
    else:
        evaluate_claude(test_questions, few_shot_examples, output_path, resume_qids)

    # 即時集計
    with open(output_path, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)
    logger.info(f"\n=== {args.model_name} 結果 ===")
    logger.info(f"全体: {correct}/{total} ({correct/total*100:.1f}%)")

    # カテゴリ別
    cat_stats = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"correct": 0, "total": 0}
        cat_stats[cat]["total"] += 1
        if r["is_correct"]:
            cat_stats[cat]["correct"] += 1
    for cat, stats in sorted(cat_stats.items()):
        acc = stats["correct"] / stats["total"] * 100
        logger.info(f"  {cat}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")


if __name__ == "__main__":
    main()
