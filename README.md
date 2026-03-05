# 日本文化理解力ベンチマーク

LLMが日本文化をどれだけ深く理解しているかを多角的に評価するベンチマークです。日本人が伝統的に受け継いできた価値観・精神性を110のキーワードに集約し、概念の理解、応用、弁別、知識の4つの軸で測定します。

## 概要

- **4軸評価**: LLMの日本文化理解力を「概念理解」「応用力」「弁別力」「知識の広さ」で多角的に測定
  - **Type A（概念理解）**: 13テーマ・100キーワードの意味を自由記述で説明させ、LLM-as-a-Judgeで採点
  - **Type B（応用力）**: 抽象概念が具体的な文化要素（茶道、歌舞伎等）にどう表れるかを問う（100問）
  - **Type C（弁別力）**: 日本文化の概念と外国文化の類似概念を比較し、日本文化の独自性を説明させる（100問）
  - **JamC-QA（知識）**: 日本固有の文化・風習に関する4択多肢選択100問でExact Match accuracy測定
- **100個の具体的文化要素**（Wikipediaから抽出）と関連付け

## 評価結果

| 略称 | モデル正式名 |
|------|-------------|
| Claude | Claude Sonnet 4.5（API） |
| GPT-OSS-Swallow | [tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1](https://huggingface.co/tokyotech-llm/GPT-OSS-Swallow-20B-RL-v0.1) |
| llm-jp v4 SFT1 | llm-jp v4-8b + LoRA SFT（未公開） |
| Qwen3-Swallow | [tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2](https://huggingface.co/tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2) |
| llm-jp v4 | llm-jp v4-8b-instruct（未公開） |
| GPT-OSS | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |
| Qwen3 | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Nemotron | [nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese) |

### 4軸総合比較

| モデル | Type A（概念理解） | Type B（応用力） | Type C（弁別力） | JamC-QA（知識） |
|--------|:-------------------:|:----------------:|:----------------:|:--------------------:|
| Claude | **15.39 / 20.0** | **17.21 / 20.0** | **16.92 / 20.0** | **89.0%** |
| GPT-OSS-Swallow | 12.16 / 20.0 | 13.73 / 20.0 | 14.67 / 20.0 | 67.0% |
| llm-jp v4 SFT1 | 12.66 / 20.0 | 13.96 / 20.0 | 12.98 / 20.0 | 61.0% |
| Qwen3-Swallow | 10.98 / 20.0 | 11.94 / 20.0 | 14.16 / 20.0 | 63.0% |
| llm-jp v4 | 12.49 / 20.0 | 12.02 / 20.0 | 11.00 / 20.0 | 58.0% |
| GPT-OSS | 11.07 / 20.0 | 11.07 / 20.0 | 13.23 / 20.0 | 46.0% |
| Qwen3 | 11.45 / 20.0 | 10.96 / 20.0 | 13.01 / 20.0 | 42.0% |
| Nemotron | 10.47 / 20.0 | 11.13 / 20.0 | 12.55 / 20.0 | 44.0% |

### Type A 観点別スコア

| 観点 | Claude | GPT-OSS-Swallow | llm-jp v4 SFT1 | Qwen3-Swallow | llm-jp v4 | GPT-OSS | Qwen3 | Nemotron |
|------|:------:|:---------------:|:---------------:|:-------------:|:---------:|:-------:|:-----:|:--------:|
| 正確性 | 4.58 | 3.75 | 3.72 | 3.34 | 3.62 | 3.43 | 3.38 | 3.12 |
| 文化的深度 | 2.88 | 2.54 | 2.70 | 2.35 | 2.54 | 2.04 | 2.23 | 2.05 |
| 要点網羅性 | 3.21 | 2.53 | 2.62 | 2.30 | 2.58 | 2.08 | 2.22 | 2.03 |
| 簡潔性 | 4.72 | 3.34 | 3.62 | 2.99 | 3.75 | 3.52 | 3.62 | 3.27 |

### Type B 観点別スコア

| 観点 | Claude | GPT-OSS-Swallow | llm-jp v4 SFT1 | Qwen3-Swallow | llm-jp v4 | GPT-OSS | Qwen3 | Nemotron |
|------|:------:|:---------------:|:---------------:|:-------------:|:---------:|:-------:|:-----:|:--------:|
| 正確性 | 4.52 | 3.58 | 3.46 | 3.10 | 3.35 | 3.02 | 3.13 | 2.81 |
| 文化的深度 | 3.64 | 2.86 | 3.08 | 2.52 | 2.46 | 2.02 | 2.18 | 2.31 |
| 場面適合性 | 4.30 | 3.35 | 3.57 | 2.89 | 2.83 | 2.67 | 2.39 | 2.62 |
| 簡潔性 | 4.75 | 3.94 | 3.85 | 3.43 | 3.38 | 3.36 | 3.26 | 3.39 |

### Type C 観点別スコア

| 観点 | Claude | GPT-OSS-Swallow | llm-jp v4 SFT1 | Qwen3-Swallow | llm-jp v4 | GPT-OSS | Qwen3 | Nemotron |
|------|:------:|:---------------:|:---------------:|:-------------:|:---------:|:-------:|:-----:|:--------:|
| 正確性 | 4.77 | 3.93 | 3.61 | 3.70 | 3.65 | 3.59 | 3.60 | 3.37 |
| 文化的深度 | 3.90 | 3.04 | 3.02 | 2.98 | 2.79 | 2.57 | 2.37 | 2.72 |
| 対比の明確さ | 4.41 | 3.62 | 3.59 | 3.41 | 3.02 | 3.07 | 3.00 | 2.99 |
| 簡潔性 | 3.84 | 4.08 | 2.76 | 4.07 | 1.54 | 4.00 | 4.04 | 3.47 |

### Type C 文化圏別スコア

| 文化圏 | Claude | GPT-OSS-Swallow | llm-jp v4 SFT1 | Qwen3-Swallow | llm-jp v4 | GPT-OSS | Qwen3 | Nemotron |
|--------|:------:|:---------------:|:---------------:|:-------------:|:---------:|:-------:|:-----:|:--------:|
| 中国 | 17.10 | 14.90 | 13.50 | 14.79 | 11.07 | 13.00 | 13.12 | 11.98 |
| 西洋 | 16.80 | 14.57 | 12.98 | 14.12 | 11.02 | 13.51 | 13.06 | 13.14 |
| 韓国 | 16.71 | 14.00 | 9.86 | 10.71 | 10.43 | 12.57 | 12.00 | 11.71 |

### JamC-QA カテゴリ別 Accuracy

| カテゴリ | Claude | GPT-OSS-Swallow | llm-jp v4 SFT1 | Qwen3-Swallow | llm-jp v4 | GPT-OSS | Qwen3 | Nemotron |
|----------|:------:|:---------------:|:---------------:|:-------------:|:---------:|:-------:|:-----:|:--------:|
| 文化 | 89.8% | 69.4% | 65.3% | 63.3% | 63.3% | 51.0% | 46.9% | 53.1% |
| 風習 | 87.8% | 65.3% | 57.1% | 61.2% | 53.1% | 38.8% | 36.7% | 36.7% |
| 風土 | 100.0% | 50.0% | 50.0% | 100.0% | 50.0% | 100.0% | 50.0% | 0.0% |

## データ構成

### 抽象的価値観（keyword.yaml）

日本人の精神性を表す14テーマ、110キーワード。

| ID | テーマ | キーワード数 | キーワード例 |
|----|--------|-------------|-------------|
| 1 | 和 | 10 | 和、配慮、謙虚、間、根回し |
| 2 | 型 | 6 | 型、作法、修練、技、匠 |
| 3 | 道 | 6 | 道、武士道、克己心、求道 |
| 4 | 気 | 8 | 気、気概、気持ち、空気 |
| 5 | 節 | 6 | 節と節目、節度、けじめ |
| 6 | 情 | 9 | 情、人情、義理、恩 |
| 7 | 忠 | 8 | 忠、上下、忠義、孝 |
| 8 | 神 | 7 | 神、禊、穢れ、大和魂 |
| 9 | 仏 | 10 | 仏、無常、悟り、禅、空 |
| 10 | 縁 | 5 | 縁、輪廻、因果 |
| 11 | 信 | 5 | 信、信用、仁、仁義 |
| 12 | 徳 | 10 | 徳、恥、礼、潔い |
| 13 | 美 | 10 | 美、わび、さび、幽玄、粋 |
| 14 | 流 | 10 | 融通、忖度、癒し、忍 |

### 具体的文化要素（keyword_validation.json）

Wikipediaから抽出した17カテゴリ、100個の文化要素。

| カテゴリ | 要素数 | 例 |
|---------|--------|-----|
| 祭と行事 | 12 | 盆踊り、祇園祭、正月、節分 |
| 食文化 | 11 | 寿司、和菓子、餅、日本酒、茶 |
| 建築 | 8 | 神社、寺院、城、庭園、茶室 |
| 武道 | 9 | 柔道、剣道、空手、相撲 |
| 芸能 | 8 | 歌舞伎、能、狂言、落語 |
| 芸道 | 4 | 茶道、華道、書道、香道 |
| 宗教 | 3 | 神道、仏教、禅 |
| 美術 | 4 | 浮世絵、水墨画、日本画 |
| 文学 | 6 | 俳句、短歌、和歌、川柳 |
| その他 | 35 | 温泉、着物、陶芸、三味線 等 |

### 文脈マッピング（context_mapping.yaml）

抽象的価値観が具体的文化要素にどのように表れるかを定義。

```yaml
# 例: テーマ「美」のキーワード「わび」
- keyword: わび
  cultural_elements:
    - 茶道
    - 茶室
    - 陶芸
    - 茶
    - 和菓子
  context: 簡素で静寂な美・深い味わい
```

- **マッピング済み**: 91個の文化要素
- **未使用**: 9個（寿司、天ぷら、そば、うどん、ラーメン、煎餅、焼酎、武士道、和楽器）

## 評価方法

### Type A: 意味説明（100問）

キーワードの意味を日本文化の文脈で説明させる。13テーマ、100キーワード。

**プロンプト形式**:
- 概念説明（テーマ=キーワード）: `日本文化の「{テーマ}」という概念を説明してください。`
- キーワード説明: `日本文化の「{テーマ}」という概念に関連して、「{キーワード}」とは何か説明してください。`

**回答制約**:
- 言語: 日本語
- 文字数: 概念説明200文字以内、キーワード説明100文字以内

**採点**: LLM-as-a-Judge（Claude Sonnet 4.5、4観点×1〜5点、満点20点）

| 観点 | 5点 | 3点 | 1点 |
|------|-----|-----|-----|
| 正確性 | 完全に正確 | 概ね正しいが一部誤り | 根本的に誤り |
| 文化的深度 | 精神性・歴史的背景を的確に捉えている | 基本的文脈は押さえるが深い洞察に欠ける | 文化的文脈が無視 |
| 要点網羅性 | 模範解答の主要要素を全て網羅 | 主要要素の半分程度をカバー | 関連性がほぼない |
| 簡潔性 | 必要十分な情報を無駄なく簡潔に | 情報の過不足あり | 構成が破綻 |

### JamC-QA: 日本文化知識（100問）

[JamC-QA](https://huggingface.co/datasets/sbintuitions/JamC-QA)（SB Intuitions）から、本プロジェクトの価値観テーマと関連の深い100問を選定。

- **形式**: 4択多肢選択
- **出典カテゴリ**: 文化（49問）、風習（49問）、風土（2問）
- **選定方法**: 1,237問（文化・風習・風土）からClaudeで関連度を評価し、上位100問を抽出
- **評価**: 4-shot, Exact Match accuracy
- **参考論文**: 岡照晃 他 (2025). JamC-QA: 日本固有の知識を問う多肢選択式質問応答ベンチマークの構築. NLP2025.

### Type B: 場面適用（100問）

抽象的キーワードが具体的な文化要素においてどう表れるかを問う。`context_mapping.yaml` の「キーワード→文化要素」マッピングから100ペアを選定。

**プロンプト形式**:
- 概念説明（テーマ=キーワード）: `日本文化の「{テーマ}」という概念（{文脈}）は、{文化要素}においてどのように表れていますか？具体的に説明してください。`
- キーワード説明: `日本文化の「{テーマ}」に関連する「{キーワード}」（{文脈}）は、{文化要素}においてどのように表れていますか？具体的に説明してください。`

**回答制約**:
- 言語: 日本語
- 文字数: 150文字以内

**採点**: LLM-as-a-Judge（Claude Sonnet 4.5、4観点×1〜5点、満点20点）

| 観点 | 5点 | 3点 | 1点 |
|------|-----|-----|-----|
| 正確性 | 完全に正確 | 概ね正しいが一部誤り | 根本的に誤り |
| 文化的深度 | 精神性・歴史的背景を的確に捉えている | 基本的文脈は押さえるが深い洞察に欠ける | 文化的文脈が無視 |
| 場面適合性 | キーワードと文化要素の関連を具体的かつ的確に説明 | 関連に触れるが具体性に欠ける | 文化要素との関連が説明されていない |
| 簡潔性 | 必要十分な情報を無駄なく簡潔に | 情報の過不足あり | 構成が破綻 |

**文化要素の選定ロジック**:
- 各キーワードから1つの文化要素を選択（先頭要素を優先）
- 同語反復回避: 仏教概念→仏教のような抽象的ペアは2番目以降の具体的要素に変更
- 集中回避: 特定の文化要素（茶道等）が過度に集中しないよう、上限を超えた場合は代替要素に変更

### Type C: 類似概念比較（100問）

日本文化の概念と外国文化（西洋・中国・韓国）の類似概念を比較し、日本文化の独自性を説明させる。`comparison_pairs.yaml` から100ペアを使用。

**プロンプト形式**:
- `日本文化の「{テーマ}」に関連する「{キーワード}」（{文脈}）と、{外国文化}文化の「{外国概念}」（{外国文脈}）は似た概念ですが、両者の違いを日本文化の特徴を踏まえて説明してください。`

**回答制約**:
- 言語: 日本語
- 文字数: 200文字以内

**採点**: LLM-as-a-Judge（Claude Sonnet 4.5、4観点×1〜5点、満点20点）

| 観点 | 5点 | 3点 | 1点 |
|------|-----|-----|-----|
| 正確性 | 両概念の説明が完全に正確 | 概ね正しいが一部に誤り | 根本的に誤り |
| 文化的深度 | 日本文化固有の精神性を的確に捉えている | 基本的文脈は押さえるが深い洞察に欠ける | 文化的文脈が無視 |
| 対比の明確さ | 共通点を踏まえ日本文化の独自性を具体的に説明 | 違いに触れるが表面的 | 対比がなく一方の説明のみ |
| 簡潔性 | 必要十分な情報を無駄なく簡潔に | 情報の過不足あり | 構成が破綻 |

**比較ペアの構成**:
- 文化圏: 西洋51問、中国42問、韓国7問
- 13テーマ × 各5〜12ペア（テーマの広さに応じて配分）
- 歴史的背景の違いにより概念が変容した例を重視

## ディレクトリ構成

```
jp_culture_benchmark/
├── README.md
├── .env                                # APIキー設定
├── keyword/
│   ├── keyword.yaml                    # 抽象的価値観（14テーマ、110キーワード）
│   └── keyword_validation.json         # 具体的文化要素（Wikipedia抽出、100個）
├── mapping/
│   ├── context_mapping.yaml            # 文脈マッピング（価値観→文化要素）
│   └── comparison_pairs.yaml           # 類似概念比較ペア（日本→外国、100ペア）
├── scripts/
│   ├── generate_type_a.py              # Type A問題生成
│   ├── generate_type_b.py              # Type B問題生成 + 模範解答下書き生成
│   ├── generate_answers.py             # vLLM OpenAI互換API経由の回答生成
│   ├── score_answers.py                # Type A 回答生成（vLLMオフライン推論）+ LLM-as-a-Judge採点
│   ├── score_type_b.py                 # Type B 回答生成（vLLM/Claude）+ LLM-as-a-Judge採点
│   ├── generate_type_c.py              # Type C問題生成 + 模範解答下書き生成
│   ├── score_type_c.py                 # Type C 回答生成（vLLM/Claude）+ LLM-as-a-Judge採点
│   ├── aggregate_results.py            # Type A 採点結果の集計・モデル間比較
│   ├── aggregate_type_b.py             # Type B 採点結果の集計・モデル間比較
│   ├── aggregate_type_c.py             # Type C 採点結果の集計・モデル間比較
│   ├── run_vllm.sh                     # vLLM環境設定ラッパー
│   ├── download_jamcqa.py              # JamC-QAデータセットDL
│   ├── select_jamcqa.py                # JamC-QAから関連100問を選定
│   ├── eval_jamcqa.py                  # JamC-QA評価（vLLM/Claude対応）
│   └── aggregate_jamcqa.py             # JamC-QA集計・モデル間比較
├── questions/
│   ├── type_a/
│   │   └── questions_claude.jsonl      # 意味説明問題（100問）
│   ├── jamcqa/
│   │   ├── dev.jsonl                   # JamC-QA devセット（32問、4-shot用）
│   │   ├── test_all.jsonl              # JamC-QA 全テスト問題（2,309問）
│   │   ├── test.jsonl                  # 選定済み100問
│   │   └── selection_log.jsonl         # 選定時のスコア・根拠ログ
│   ├── type_b/
│   │   └── questions.jsonl             # 場面適用問題（100問）
│   └── type_c/
│       └── questions.jsonl             # 類似概念比較問題（100問）
├── answers/
│   ├── type_a/
│   │   ├── answers_human.jsonl         # 人間の専門家による模範解答（100問）
│   │   ├── answers_claude.jsonl        # Claude Sonnet 4.5による回答
│   │   ├── answers_qwen3-8b.jsonl      # Qwen3-8Bによる回答
│   │   ├── answers_llmjp-v4-8b.jsonl   # llm-jp v4-8bによる回答
│   │   └── answers_gpt-oss-20b.jsonl   # GPT-OSS-20Bによる回答
│   ├── type_b/
│   │   ├── answers_human_draft.jsonl   # 模範解答下書き（Claude生成）
│   │   ├── answers_human.jsonl         # 模範解答（人手修正後の最終版）
│   │   ├── answers_claude.jsonl        # Claude Sonnet 4.5による回答
│   │   ├── answers_qwen3-8b.jsonl      # Qwen3-8Bによる回答
│   │   ├── answers_llmjp-v4-8b.jsonl   # llm-jp v4-8bによる回答
│   │   └── answers_gpt-oss-20b.jsonl   # GPT-OSS-20Bによる回答
│   └── type_c/
│       ├── answers_human_draft.jsonl   # 模範解答下書き（Claude生成）
│       ├── answers_human.jsonl         # 模範解答（人手修正後の最終版）
│       ├── answers_claude.jsonl        # Claude Sonnet 4.5による回答
│       ├── answers_qwen3-8b.jsonl      # Qwen3-8Bによる回答
│       ├── answers_llmjp-v4-8b.jsonl   # llm-jp v4-8bによる回答
│       └── answers_gpt-oss-20b.jsonl   # GPT-OSS-20Bによる回答
└── results/
    ├── type_a/
    │   ├── scores_{model}.jsonl        # Type A 問題ごとの採点結果
    │   ├── summary_{model}.json        # Type A モデル別集計
    │   └── comparison.json             # Type A モデル間比較
    ├── type_b/
    │   ├── scores_{model}.jsonl        # Type B 問題ごとの採点結果
    │   ├── summary_{model}.json        # Type B モデル別集計
    │   └── comparison.json             # Type B モデル間比較
    ├── type_c/
    │   ├── scores_{model}.jsonl        # Type C 問題ごとの採点結果
    │   ├── summary_{model}.json        # Type C モデル別集計
    │   └── comparison.json             # Type C モデル間比較
    └── jamcqa/
        ├── answers_{model}.jsonl       # JamC-QA 回答・正誤結果
        ├── summary_{model}.json        # JamC-QA モデル別集計
        └── comparison.json             # JamC-QA モデル間比較
```

## 使い方

### Type A: 回答生成 + 採点

```bash
# vLLM環境で回答生成 + 採点
bash scripts/run_vllm.sh score_answers.py \
    --model_name qwen3-8b --model_path Qwen/Qwen3-8B

# 回答生成のみ（採点スキップ）
bash scripts/run_vllm.sh score_answers.py \
    --model_name qwen3-8b --model_path Qwen/Qwen3-8B --generate_only

# 既存回答の採点のみ
python scripts/score_answers.py --model_name claude

# 集計・比較
python scripts/aggregate_results.py --all
```

### Type B: 回答生成 + 採点

```bash
# 問題生成（初回のみ）
python scripts/generate_type_b.py --questions_only

# 問題生成 + 模範解答下書き生成（Claude SDK）
python scripts/generate_type_b.py

# vLLM環境で回答生成 + 採点
bash scripts/run_vllm.sh score_type_b.py \
    --model_name qwen3-8b --model_path Qwen/Qwen3-8B

# Claude回答生成 + 採点（回答ファイルがなければ自動生成）
python scripts/score_type_b.py --model_name claude

# 集計・比較
python scripts/aggregate_type_b.py --all
```

### Type C: 回答生成 + 採点

```bash
# 問題生成（初回のみ）
python scripts/generate_type_c.py --questions_only

# 問題生成 + 模範解答下書き生成（Claude SDK）
python scripts/generate_type_c.py

# vLLM環境で回答生成 + 採点
bash scripts/run_vllm.sh score_type_c.py \
    --model_name qwen3-8b --model_path Qwen/Qwen3-8B

# Claude回答生成 + 採点（回答ファイルがなければ自動生成）
python scripts/score_type_c.py --model_name claude

# 集計・比較
python scripts/aggregate_type_c.py --all
```

### JamC-QA: 多肢選択評価

```bash
# データセットDL（初回のみ）
/home/llm-user/envs/local_llm/bin/python scripts/download_jamcqa.py

# 100問選定（初回のみ）
python scripts/select_jamcqa.py

# ローカルモデル評価
bash scripts/run_vllm.sh eval_jamcqa.py \
    --model_name qwen3-8b --model_path Qwen/Qwen3-8B

# Claude評価
python scripts/eval_jamcqa.py --model_name claude

# 集計・比較
python scripts/aggregate_jamcqa.py --all
```

## 進捗状況

- [x] テーマ・キーワードの定義（keyword.yaml）
- [x] 具体的文化要素の抽出（keyword_validation.json）
- [x] 文脈マッピングの作成（context_mapping.yaml）
- [x] 文脈マッピングの調整（91/100要素使用）
- [x] Type A: 意味説明問題の作成（100問）
- [x] Type A: 人間の模範解答の作成（100問）
- [x] Type A: 8モデルの回答生成・採点・集計完了
- [x] JamC-QA: データセットDL・関連100問の選定
- [x] JamC-QA: 8モデルの評価完了
- [x] Type B: 場面適用問題の作成（100問）
- [x] Type B: 人間の模範解答の作成（Claude下書き → 人手修正）
- [x] Type B: 8モデルの回答生成・採点・集計完了
- [x] Type C: 類似概念比較ペアの定義（comparison_pairs.yaml、100ペア）
- [x] Type C: 問題生成（100問）
- [x] Type C: 人間の模範解答の作成（Claude下書き → 人手修正）
- [x] Type C: 8モデルの回答生成・採点・集計完了
- [x] 4軸総合比較の実施（8モデル）

## 出典

- 山久瀬洋二. (2023). 日英対訳 日本人のこころ 増補改訂版 (M. A. Cooney, Trans.). IBCパブリッシング.
- 岡照晃, 柴田知秀, 吉田奈央. (2025). JamC-QA: 日本固有の知識を問う多肢選択式質問応答ベンチマークの構築. 言語処理学会第31回年次大会(NLP2025).
