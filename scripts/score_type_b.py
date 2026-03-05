#!/usr/bin/env python3
"""
Type B: 場面適用問題の回答生成 + LLM-as-a-Judge 採点スクリプト。

vLLMオフライン推論で回答を生成し、Claude Code SDKで採点する。

使い方:
    # 既存の回答ファイルを採点
    python scripts/score_type_b.py --model_name claude

    # vLLMで回答生成してから採点
    python scripts/score_type_b.py --model_name qwen3-8b --model_path Qwen/Qwen3-8B
    python scripts/score_type_b.py --model_name llmjp-v4-8b \
        --model_path /home/llm-user/datadrive/llm-jp-4/v4-8b-decay2m-ipt_v3.1-instruct4/

    # 回答生成のみ（採点スキップ）
    python scripts/score_type_b.py --model_name qwen3-8b --model_path Qwen/Qwen3-8B --generate_only

    # 途中再開
    python scripts/score_type_b.py --model_name qwen3-8b --resume

実行環境:
    vLLM使用時は local_llm 環境で実行してください:
    bash scripts/run_vllm.sh score_type_b.py --model_name qwen3-8b --model_path Qwen/Qwen3-8B
"""

import argparse
import json
import os
import re
import subprocess
import time
import logging
from datetime import datetime, timezone
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 設定
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
CRITERIA = ["正確性", "文化的深度", "場面適合性", "簡潔性"]
RETRY_MAX = 3
RETRY_DELAY = 5

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

def _check_system_prompt_support(tokenizer):
    """チャットテンプレートがsystemプロンプトの内容を保持するか確認する。"""
    test_marker = "__SYSTEM_CONTENT_TEST__"
    messages = [
        {"role": "system", "content": test_marker},
        {"role": "user", "content": "test"},
    ]
    try:
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        supported = test_marker in result
        if not supported:
            logger.warning(
                "このモデルのチャットテンプレートはsystemプロンプトの内容を無視します。"
                "systemの指示をuserメッセージに埋め込みます。"
            )
        return supported
    except Exception:
        return True


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_FILE = os.path.join(BASE_DIR, "questions", "type_b", "questions.jsonl")
ANSWERS_DIR = os.path.join(BASE_DIR, "answers", "type_b")
HUMAN_ANSWERS_FILE = os.path.join(ANSWERS_DIR, "answers_human.jsonl")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "type_b")


# ============================================================
# 回答生成（vLLMオフライン推論）
# ============================================================

def generate_answers_vllm(model_path, questions, output_file):
    """vLLMオフライン推論で全問の回答を一括生成する。"""
    from vllm import LLM, SamplingParams

    # 生成済みIDを取得（レジューム対応）
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["id"])

    pending = [q for q in questions if q["id"] not in existing_ids]
    if not pending:
        logger.info("全問生成済みです。")
        return

    logger.info(f"回答生成: {len(pending)}問 (モデル: {model_path})")

    # vLLMモデル読み込み
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )

    # システムプロンプトのサポート確認
    supports_system = _check_system_prompt_support(tokenizer)

    # プロンプト構築
    system_msg = (
        "あなたは日本文化に精通した専門家です。"
        "質問に日本語で簡潔に答えてください。回答は150文字以内としてください。"
    )
    prompts = []
    for q in pending:
        if supports_system:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": q["prompt"]},
            ]
        else:
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": f"{system_msg}\n\n{q['prompt']}"},
            ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt_text)

    # 一括推論
    logger.info("vLLM推論開始...")
    outputs = llm.generate(prompts, sampling_params)

    # 結果保存
    with open(output_file, "a", encoding="utf-8") as out:
        for q, output in zip(pending, outputs):
            answer_text = strip_think_tags(output.outputs[0].text)
            record = {
                "id": q["id"],
                "theme_name": q["theme_name"],
                "keyword": q["keyword"],
                "cultural_element": q["cultural_element"],
                "prompt": q["prompt"],
                "answer": answer_text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"回答生成完了: {output_file}")

    # GPU メモリ解放
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================
# 回答生成（Claude Code SDK）
# ============================================================

ANSWER_MODEL = "claude-sonnet-4-5-20250929"

def generate_answers_claude(questions, output_file):
    """Claude Code SDKで全問の回答を生成する。"""
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_ids.add(json.loads(line)["id"])

    pending = [q for q in questions if q["id"] not in existing_ids]
    if not pending:
        logger.info("全問生成済みです。")
        return

    logger.info(f"Claude回答生成: {len(pending)}問")

    system_msg = (
        "あなたは日本文化に精通した専門家です。"
        "質問に日本語で簡潔に答えてください。回答は150文字以内としてください。"
    )

    with open(output_file, "a", encoding="utf-8") as out:
        for q in tqdm(pending, desc="Claude回答生成中"):
            prompt_text = f"{system_msg}\n\n{q['prompt']}"

            for attempt in range(RETRY_MAX):
                try:
                    answer_text = call_claude_sdk(prompt_text, model=ANSWER_MODEL)
                    break
                except (subprocess.TimeoutExpired, RuntimeError) as e:
                    logger.warning(f"エラー ({q['id']}): {e}, リトライ {attempt + 1}/{RETRY_MAX}")
                    if attempt < RETRY_MAX - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.error(f"回答生成失敗 ({q['id']}): {e}")
                        answer_text = ""

            record = {
                "id": q["id"],
                "theme_name": q["theme_name"],
                "keyword": q["keyword"],
                "cultural_element": q["cultural_element"],
                "prompt": q["prompt"],
                "answer": answer_text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    logger.info(f"Claude回答生成完了: {output_file}")


# ============================================================
# 採点（Claude Code SDK）
# ============================================================

def build_judge_prompt(theme_name, keyword, cultural_element, prompt, reference_answer, target_answer):
    return f"""あなたは日本文化の専門家として、LLMによる日本文化の場面適用に関する回答を評価する審査員です。

## 評価対象
- テーマ: {theme_name}
- キーワード: {keyword}
- 文化要素: {cultural_element}
- 質問: {prompt}
- 参照回答（人間の専門家による模範解答）:
{reference_answer}
- 評価対象の回答:
{target_answer}

## 評価基準
以下の4つの観点でそれぞれ1〜5点で採点してください。

### 1. 正確性（せいかくせい）
キーワードと文化要素の関連に関する説明が事実として正しいかどうかを評価します。
- 5点: 事実に基づいた完全に正確な説明である
- 4点: ほぼ正確だが、軽微な不正確さや曖昧さがある
- 3点: 概ね正しいが、一部に誤りや誤解を招く表現がある
- 2点: 重要な事実誤認や不正確な説明が含まれる
- 1点: 根本的に誤った説明である、または質問に答えていない

### 2. 文化的深度（ぶんかてきしんど）
日本文化特有のニュアンスや精神性をどれだけ深く捉えているかを評価します。
- 5点: 日本文化の深層にある精神性や歴史的背景を的確に捉え、表面的でない深い理解を示している
- 4点: 文化的な背景や関連概念にも触れ、一定の深さがある
- 3点: 基本的な文化的文脈は押さえているが、深い洞察には欠ける
- 2点: 文化的背景への言及が乏しく、表面的な説明にとどまる
- 1点: 文化的な文脈がほとんど無視されている

### 3. 場面適合性（ばめんてきごうせい）
キーワードが文化要素においてどのように表れるか、具体的かつ的確に説明できているかを評価します。
- 5点: キーワードが文化要素にどう表れるか、具体的な事例や実践を挙げて的確に説明している
- 4点: 関連は明確だが、具体的な事例がやや不足
- 3点: 関連に触れているが、一般的・抽象的な説明にとどまる
- 2点: 文化要素への言及はあるが、キーワードとの具体的な結びつきが希薄
- 1点: 文化要素との関連がほとんど説明されていない

### 4. 簡潔性（かんけつせい）
制約条件の範囲内で、無駄なく適切にまとめられているかを評価します。
- 5点: 必要十分な情報を無駄なく簡潔にまとめている
- 4点: 概ね簡潔だが、やや冗長な部分や不足がある
- 3点: 情報の過不足があり、構成に改善の余地がある
- 2点: 冗長で焦点が定まらない、または極端に情報が不足している
- 1点: 構成が破綻している、または意味をなさない

## 回答形式
必ず以下のJSON形式のみで回答してください。JSON以外のテキストは含めないでください。

```json
{{
  "正確性": {{"score": <1-5の整数>, "reason": "<日本語で1-2文の根拠>"}},
  "文化的深度": {{"score": <1-5の整数>, "reason": "<日本語で1-2文の根拠>"}},
  "場面適合性": {{"score": <1-5の整数>, "reason": "<日本語で1-2文の根拠>"}},
  "簡潔性": {{"score": <1-5の整数>, "reason": "<日本語で1-2文の根拠>"}}
}}
```"""


def parse_judge_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
        for criterion in CRITERIA:
            assert criterion in parsed, f"Missing criterion: {criterion}"
            assert "score" in parsed[criterion], f"Missing score for {criterion}"
            assert "reason" in parsed[criterion], f"Missing reason for {criterion}"
            score = parsed[criterion]["score"]
            assert isinstance(score, int) and 1 <= score <= 5, f"Invalid score: {score}"
        return parsed
    except (json.JSONDecodeError, AssertionError, KeyError) as e:
        logger.warning(f"JSON解析失敗 ({e})、正規表現で抽出を試行")

    result = {}
    for criterion in CRITERIA:
        pattern = (
            rf'"{re.escape(criterion)}"\s*:\s*\{{\s*"score"\s*:\s*(\d+)\s*,\s*"reason"\s*:\s*"([^"]+)"'
        )
        match = re.search(pattern, text)
        if match:
            result[criterion] = {"score": int(match.group(1)), "reason": match.group(2)}
    if len(result) == len(CRITERIA):
        return result
    raise ValueError(f"応答の解析に失敗: {text[:200]}")


def call_claude_sdk(prompt_text, model=JUDGE_MODEL):
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


def evaluate_single(question, reference, target_answer):
    prompt_text = build_judge_prompt(
        theme_name=question["theme_name"],
        keyword=question["keyword"],
        cultural_element=question["cultural_element"],
        prompt=question["prompt"],
        reference_answer=reference["answer"],
        target_answer=target_answer,
    )

    for attempt in range(RETRY_MAX):
        try:
            response_text = call_claude_sdk(prompt_text)
            return parse_judge_response(response_text)
        except subprocess.TimeoutExpired:
            logger.warning(f"タイムアウト、リトライ {attempt + 1}/{RETRY_MAX}")
            time.sleep(RETRY_DELAY)
        except RuntimeError as e:
            logger.error(f"CLIエラー: {e}")
            if attempt < RETRY_MAX - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise
        except ValueError as e:
            logger.error(f"解析エラー: {e}")
            if attempt < RETRY_MAX - 1:
                time.sleep(1)
            else:
                raise
    raise RuntimeError("最大リトライ回数を超過")


# ============================================================
# メイン
# ============================================================

def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                data[record["id"]] = record
    return data


def load_jsonl_list(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Type B: 回答生成 + LLM-as-a-Judge 採点")
    parser.add_argument("--model_name", required=True, help="モデル名（ファイル名に使用）")
    parser.add_argument("--model_path", default=None,
                        help="vLLMモデルパス（指定時は回答生成を実行）")
    parser.add_argument("--generate_only", action="store_true",
                        help="回答生成のみ（採点スキップ）")
    parser.add_argument("--resume", action="store_true", help="途中から再開")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ANSWERS_DIR, exist_ok=True)

    questions_list = load_jsonl_list(QUESTIONS_FILE)
    questions = {q["id"]: q for q in questions_list}
    answers_file = os.path.join(ANSWERS_DIR, f"answers_{args.model_name}.jsonl")

    # ── Step 1: 回答生成 ──
    if args.model_path:
        # vLLMで回答生成
        generate_answers_vllm(args.model_path, questions_list, answers_file)
        if args.generate_only:
            return
    elif not os.path.exists(answers_file):
        # model_pathなし & 回答ファイルなし → Claude SDKで生成
        logger.info(f"回答ファイルが見つかりません。Claude SDKで回答を生成します。")
        generate_answers_claude(questions_list, answers_file)
        if args.generate_only:
            return

    # ── Step 2: 採点 ──
    if not os.path.exists(answers_file):
        logger.error(f"回答ファイルが見つかりません: {answers_file}")
        logger.error("--model_path を指定して回答を生成してください。")
        return

    references = load_jsonl(HUMAN_ANSWERS_FILE)
    targets = load_jsonl(answers_file)
    output_file = os.path.join(RESULTS_DIR, f"scores_{args.model_name}.jsonl")

    # レジューム
    scored_ids = set()
    if args.resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    scored_ids.add(json.loads(line)["id"])
        logger.info(f"レジューム: {len(scored_ids)}件は採点済み")

    pending = [qid for qid in sorted(questions.keys()) if qid not in scored_ids]
    if not pending:
        print("全問採点済みです。")
        return

    print(f"採点対象: {len(pending)}問 (モデル: {args.model_name})")
    print(f"審査員: {JUDGE_MODEL} (Claude Code SDK)")

    with open(output_file, "a", encoding="utf-8") as out:
        for qid in tqdm(pending, desc=f"採点中 ({args.model_name})"):
            q = questions[qid]
            ref = references[qid]
            target = targets.get(qid)
            if target is None:
                logger.warning(f"回答が見つかりません: {qid}")
                continue

            try:
                scores = evaluate_single(q, ref, strip_think_tags(target["answer"]))
            except Exception as e:
                logger.error(f"採点失敗 ({qid}): {e}")
                continue

            total = sum(s["score"] for s in scores.values())
            avg = total / len(CRITERIA)

            record = {
                "id": qid,
                "theme_id": q["theme_id"],
                "theme_name": q["theme_name"],
                "keyword": q["keyword"],
                "cultural_element": q["cultural_element"],
                "scores": scores,
                "total_score": total,
                "average_score": round(avg, 2),
                "target_answer": target["answer"],
                "reference_answer": ref["answer"],
                "judge_model": JUDGE_MODEL,
                "judged_at": datetime.now(timezone.utc).isoformat(),
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    print(f"採点完了: {output_file}")


if __name__ == "__main__":
    main()
