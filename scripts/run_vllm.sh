#!/bin/bash
# vLLMオフライン推論用の環境設定 + スクリプト実行
#
# 使い方:
#   bash scripts/run_vllm.sh score_answers.py --model_name qwen3-8b --model_path Qwen/Qwen3-8B
#   bash scripts/run_vllm.sh eval_jamcqa.py --model_name qwen3-8b --model_path Qwen/Qwen3-8B
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# 第1引数をスクリプト名として取得（デフォルト: score_answers.py）
SCRIPT="${1:-score_answers.py}"
shift

# local_llm 環境
VENV=/home/llm-user/envs/local_llm

export CUDA_HOME="$VENV"
export PATH="$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$VENV/lib:${LIBRARY_PATH:-}"

exec "$VENV/bin/python" "$SCRIPT_DIR/$SCRIPT" "$@"
