#!/usr/bin/env python3
"""Generate docs/analysis.ipynb - comprehensive analysis notebook for JP Culture Benchmark."""
import json
import os

cells = []

def md(source):
    """Add a markdown cell."""
    lines = source.strip().split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": src,
    })

def code(source):
    """Add a code cell."""
    lines = source.strip().split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None,
    })

# ==============================================================================
# Cell 0: Title
# ==============================================================================
md("""# 日本文化理解力ベンチマーク 分析レポート

**Japanese Culture Understanding Benchmark — Analysis Report**

---

## 目次

1. はじめに
2. ベンチマークの設計
3. llm-jp v4の評価結果
   - 3.1 総合評価
   - 3.2 Type A（概念理解）
   - 3.3 Type B（応用力）
   - 3.4 Type C（弁別力）
   - 3.5 JamC-QA（知識）
   - 3.6 軸横断分析
4. ベンチマークの有効性検証
   - 4.1 軸間の独立性
   - 4.2 モデル弁別力
   - 4.3 評価基準の妥当性
   - 4.4 問題設計の適切性
5. 含意
   - 5.1 日本文化理解の現状
   - 5.2 モデルアーキテクチャの影響
   - 5.3 SFTの改善可能性
6. 課題と限界
7. まとめ""")

# ==============================================================================
# Cell 1: Setup & Data Loading
# ==============================================================================
code("""import json
import os
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ---------- Project Root ----------
PROJECT_ROOT = Path('/home/llm-user/jp_culture_benchmark')
os.chdir(PROJECT_ROOT)

# ---------- Japanese Font Setup ----------
# Register IPAexGothic font directly
_font_path = '/home/llm-user/miniconda3/lib/python3.13/site-packages/japanize_matplotlib/fonts/ipaexg.ttf'
if os.path.exists(_font_path):
    fm.fontManager.addfont(_font_path)
    matplotlib.rcParams['font.family'] = 'IPAexGothic'
    print(f"Font registered: IPAexGothic from {_font_path}")
else:
    import japanize_matplotlib
    print("Using japanize_matplotlib fallback")

# ---------- Style ----------
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'IPAexGothic'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize'] = (10, 6)

FIGDIR = PROJECT_ROOT / 'docs' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------- Constants ----------
MODEL_IDS = [
    'claude', 'llmjp-v4-8b', 'llm-jp-4-swallow-sft1',
    'qwen3-8b', 'qwen3-swallow-8b',
    'gpt-oss-20b', 'gpt-oss-swallow-20b',
    'nemotron-nano-9b',
]

MODEL_DISPLAY = {
    'claude': 'Claude Sonnet 4.5',
    'llmjp-v4-8b': 'llm-jp v4 8B',
    'llm-jp-4-swallow-sft1': 'llm-jp v4 SFT1',
    'qwen3-8b': 'Qwen3 8B',
    'qwen3-swallow-8b': 'Qwen3-Swallow 8B',
    'gpt-oss-20b': 'GPT-OSS 20B',
    'gpt-oss-swallow-20b': 'GPT-OSS-Swallow 20B',
    'nemotron-nano-9b': 'Nemotron Nano 9B',
}

MODEL_SHORT = {
    'claude': 'Claude',
    'llmjp-v4-8b': 'llm-jp v4',
    'llm-jp-4-swallow-sft1': 'llm-jp SFT1',
    'qwen3-8b': 'Qwen3',
    'qwen3-swallow-8b': 'Qwen3-SW',
    'gpt-oss-20b': 'GPT-OSS',
    'gpt-oss-swallow-20b': 'GPT-OSS-SW',
    'nemotron-nano-9b': 'Nemotron',
}

MODEL_COLORS = {
    'claude': '#e74c3c',
    'llmjp-v4-8b': '#2196F3',
    'llm-jp-4-swallow-sft1': '#1565C0',
    'qwen3-8b': '#FF9800',
    'qwen3-swallow-8b': '#E65100',
    'gpt-oss-20b': '#4CAF50',
    'gpt-oss-swallow-20b': '#2E7D32',
    'nemotron-nano-9b': '#9E9E9E',
}

SFT_PAIRS = [
    ('llmjp-v4-8b', 'llm-jp-4-swallow-sft1', 'llm-jp v4 → SFT1'),
    ('qwen3-8b', 'qwen3-swallow-8b', 'Qwen3 → Swallow'),
    ('gpt-oss-20b', 'gpt-oss-swallow-20b', 'GPT-OSS → Swallow'),
]

AXES = ['type_a', 'type_b', 'type_c', 'jamcqa']
AXIS_DISPLAY = {
    'type_a': 'Type A\\n(概念理解)',
    'type_b': 'Type B\\n(応用力)',
    'type_c': 'Type C\\n(弁別力)',
    'jamcqa': 'JamC-QA\\n(知識)',
}
AXIS_SHORT = {
    'type_a': 'Type A',
    'type_b': 'Type B',
    'type_c': 'Type C',
    'jamcqa': 'JamC-QA',
}

CRITERIA_A = ['正確性', '文化的深度', '要点網羅性', '簡潔性']
CRITERIA_B = ['正確性', '文化的深度', '場面適合性', '簡潔性']
CRITERIA_C = ['正確性', '文化的深度', '対比の明確さ', '簡潔性']
CRITERIA_BY_TYPE = {'type_a': CRITERIA_A, 'type_b': CRITERIA_B, 'type_c': CRITERIA_C}

CULTURES = ['中国', '西洋', '韓国']

THEMES = ['和', '型', '道', '気', '節', '情', '忠', '神', '仏', '縁', '信', '徳', '美']

# ---------- Data Loading ----------
def load_summary(axis, model):
    path = f'results/{axis}/summary_{model}.json'
    with open(path) as f:
        return json.load(f)

def load_scores(axis, model):
    path = f'results/{axis}/scores_{model}.jsonl'
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def load_jamcqa_answers(model):
    path = f'results/jamcqa/answers_{model}.jsonl'
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

# Load all summaries
summaries = {}
for axis in AXES:
    summaries[axis] = {}
    for m in MODEL_IDS:
        summaries[axis][m] = load_summary(axis, m)

# Load all per-question scores into DataFrames
scores_dfs = {}
for axis in ['type_a', 'type_b', 'type_c']:
    frames = []
    for m in MODEL_IDS:
        rows = load_scores(axis, m)
        for r in rows:
            flat = {'model': m, 'id': r['id'], 'theme_name': r['theme_name'],
                    'keyword': r['keyword'], 'total_score': r['total_score']}
            if 'cultural_element' in r:
                flat['cultural_element'] = r['cultural_element']
            if 'foreign_concept' in r:
                flat['foreign_concept'] = r['foreign_concept']
            if 'foreign_culture' in r:
                flat['foreign_culture'] = r['foreign_culture']
            for crit, val in r['scores'].items():
                flat[crit] = val['score']
            frames.append(flat)
    scores_dfs[axis] = pd.DataFrame(frames)

# Load JamCQA answers
jamcqa_frames = []
for m in MODEL_IDS:
    rows = load_jamcqa_answers(m)
    for r in rows:
        r['model'] = m
        jamcqa_frames.append(r)
jamcqa_df = pd.DataFrame(jamcqa_frames)

print("Data loaded successfully!")
print(f"Type A: {len(scores_dfs['type_a'])} records")
print(f"Type B: {len(scores_dfs['type_b'])} records")
print(f"Type C: {len(scores_dfs['type_c'])} records")
print(f"JamC-QA: {len(jamcqa_df)} records")""")

# ==============================================================================
# Cell 2: Chapter 1 - Introduction
# ==============================================================================
md("""## 第1章: はじめに

### 背景と目的

大規模言語モデル（LLM）の日本語能力は急速に向上しているが、**日本文化に対する深い理解力**を体系的に測定するベンチマークは存在しなかった。既存の日本語ベンチマークは主に言語知識・推論能力を測定するものであり、文化的文脈の理解や文化間比較の能力は測定対象外であった。

本ベンチマークは、LLMの日本文化理解力を**4つの独立した軸**で測定する：

| 軸 | 測定対象 | 問題形式 | 問題数 |
|---|---------|---------|-------|
| **Type A** | 概念理解 | 自由記述（200字以内） | 100 |
| **Type B** | 応用力 | 自由記述（150字以内） | 100 |
| **Type C** | 弁別力 | 自由記述（200字以内） | 100 |
| **JamC-QA** | 文化知識 | 4択（4-shot） | 100 |

### 評価対象モデル（8モデル）""")

# ==============================================================================
# Cell 3: Model table
# ==============================================================================
code("""# Model overview table
model_info = pd.DataFrame([
    ['Claude Sonnet 4.5', 'claude', 'Anthropic', '非公開', 'API・ベースライン（上限参考）'],
    ['llm-jp v4 8B', 'llmjp-v4-8b', 'LLM-jp', '8B', 'ベースモデル（instruct版）'],
    ['llm-jp v4 SFT1', 'llm-jp-4-swallow-sft1', 'LLM-jp/Swallow', '8B+LoRA', 'SFT後モデル'],
    ['Qwen3 8B', 'qwen3-8b', 'Alibaba', '8B', 'マルチリンガルベース'],
    ['Qwen3-Swallow 8B', 'qwen3-swallow-8b', 'Swallow/Tokyo Tech', '8B', 'Swallow RL適用'],
    ['GPT-OSS 20B', 'gpt-oss-20b', 'OpenAI', '20B', '英語主体ベース'],
    ['GPT-OSS-Swallow 20B', 'gpt-oss-swallow-20b', 'Swallow/Tokyo Tech', '20B', 'Swallow RL適用'],
    ['Nemotron Nano 9B', 'nemotron-nano-9b', 'NVIDIA', '9B', '日本語対応'],
], columns=['モデル名', 'ID', '開発元', 'パラメータ数', '特徴'])

print(model_info.to_markdown(index=False))""")

# ==============================================================================
# Cell 4: Chapter 2 - Benchmark Design
# ==============================================================================
md("""## 第2章: ベンチマークの設計

### 4軸設計の哲学

本ベンチマークは「文化理解」を多角的に捉えるために、4つの独立した評価軸を設計した：

1. **Type A（概念理解）**: 日本文化の抽象概念（「和」「道」「型」など）をどれだけ正確かつ深く説明できるか
2. **Type B（応用力）**: 抽象概念が具体的な文化要素（畳、茶道、相撲など）にどう表れるか説明できるか
3. **Type C（弁別力）**: 日本の概念と外国の類似概念（中国・西洋・韓国）の違いを的確に弁別できるか
4. **JamC-QA（文化知識）**: 日本文化に関する事実的知識を正確に保持しているか

### 問題構成

- **13テーマ**: 和・型・道・気・節・情・忠・神・仏・縁・信・徳・美
- **100キーワード**: 各テーマに5〜10のキーワード
- **100文化要素**: Wikipedia検証済みの具体的文化要素（Type B用）
- **100比較ペア**: 中国42件・西洋51件・韓国7件（Type C用）

### LLM-as-a-Judge 方法論

Type A/B/Cは**Claude Sonnet 4.5**によるLLM-as-a-Judge方式で評価。各問題に対し、人間が作成した**参照回答**と比較して4観点で1〜5点の採点を行う。

| 評価軸 | 観点1 | 観点2 | 観点3 | 観点4 |
|-------|-------|-------|-------|-------|
| Type A | 正確性 | 文化的深度 | 要点網羅性 | 簡潔性 |
| Type B | 正確性 | 文化的深度 | 場面適合性 | 簡潔性 |
| Type C | 正確性 | 文化的深度 | 対比の明確さ | 簡潔性 |""")

# ==============================================================================
# Cell 5: Chapter 3 intro
# ==============================================================================
md("""## 第3章: llm-jp v4の評価結果

### 3.1 総合評価""")

# ==============================================================================
# Cell 6: Fig01 + Fig02 — Overall comparison
# ==============================================================================
code("""# === Fig 01: Overall Comparison — Bar (Type A-C) + Line (JamC-QA on 2nd axis) ===

# Compute scores for each model/axis
overall_scores = {}
for m in MODEL_IDS:
    overall_scores[m] = {}
    for axis in AXES:
        s = summaries[axis][m]
        if axis == 'jamcqa':
            overall_scores[m][axis] = s['overall']['accuracy'] * 100
        else:
            overall_scores[m][axis] = s['overall']['average_total']

fig, ax1 = plt.subplots(figsize=(14, 7))
x = np.arange(len(MODEL_IDS))
width = 0.25
bar_axes = ['type_a', 'type_b', 'type_c']
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, axis in enumerate(bar_axes):
    vals = [overall_scores[m][axis] for m in MODEL_IDS]
    ax1.bar(x + i * width - width, vals, width,
            label=AXIS_SHORT[axis], color=colors_bar[i], alpha=0.85)

ax1.set_ylabel('スコア（Type A-C, /20点満点）')
ax1.set_ylim(0, 22)
ax1.axhline(y=20, color='gray', linestyle='--', alpha=0.2)
ax1.set_xticks(x)
ax1.set_xticklabels([MODEL_SHORT[m] for m in MODEL_IDS], rotation=30, ha='right')

# Secondary y-axis: JamC-QA accuracy as line
ax2 = ax1.twinx()
jamcqa_vals = [overall_scores[m]['jamcqa'] for m in MODEL_IDS]
ax2.plot(x, jamcqa_vals, color='#d62728', marker='D', markersize=8,
         linewidth=2.5, label='JamC-QA（精度%）', zorder=5)
ax2.set_ylabel('JamC-QA 精度（%）')
ax2.set_ylim(0, 105)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

ax1.set_title('図1: 8モデル × 4軸 総合比較')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig01_overall_comparison.png')
plt.show()
print("Fig01 saved.")""")

# ==============================================================================
# Cell 7: Fig02 — Radar chart
# ==============================================================================
code("""# === Fig 02: llm-jp v4 / SFT1 / Claude 4-axis Radar Chart ===

def radar_chart(ax, labels, values_dict, title, max_val=None):
    \"\"\"Draw a radar chart.\"\"\"
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    plt.xticks(angles[:-1], labels, fontsize=10)

    for name, vals in values_dict.items():
        v = vals + vals[:1]
        color = MODEL_COLORS.get(name, '#333')
        ax.plot(angles, v, 'o-', linewidth=2, label=MODEL_SHORT.get(name, name),
                color=color)
        ax.fill(angles, v, alpha=0.1, color=color)

    if max_val:
        ax.set_ylim(0, max_val)
    ax.set_title(title, pad=20, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Prepare data
radar_models = ['llmjp-v4-8b', 'llm-jp-4-swallow-sft1', 'claude']
radar_labels = ['Type A\\n(概念理解)', 'Type B\\n(応用力)', 'Type C\\n(弁別力)', 'JamC-QA\\n(知識)']
radar_data = {}
for m in radar_models:
    vals = []
    for axis in AXES:
        s = summaries[axis][m]
        if axis == 'jamcqa':
            vals.append(s['overall']['accuracy'] * 20)  # Normalize to 0-20 scale
        else:
            vals.append(s['overall']['average_total'])
    radar_data[m] = vals

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
radar_chart(ax, radar_labels, radar_data, '図2: llm-jp v4 / SFT1 / Claude\\n4軸レーダーチャート', max_val=20)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig02_llmjp_radar_4axis.png')
plt.show()
print("Fig02 saved.")""")

# ==============================================================================
# Cell 8: 3.1 Overall ranking table
# ==============================================================================
code("""# Overall Ranking Table
print("### 総合ランキング\\n")
ranking_data = []
for m in MODEL_IDS:
    row = {'モデル': MODEL_DISPLAY[m]}
    total = 0
    for axis in AXES:
        s = summaries[axis][m]
        if axis == 'jamcqa':
            val = s['overall']['accuracy'] * 100
            row['JamC-QA (%)'] = f"{val:.1f}"
        else:
            val = s['overall']['average_total']
            row[AXIS_SHORT[axis] + ' (/20)'] = f"{val:.2f}"
        total += val if axis != 'jamcqa' else s['overall']['accuracy'] * 20
    row['合計 (/80)'] = f"{total:.2f}"
    ranking_data.append(row)

ranking_df = pd.DataFrame(ranking_data)
ranking_df = ranking_df.sort_values('合計 (/80)', ascending=False)
print(ranking_df.to_markdown(index=False))""")

# ==============================================================================
# Cell 9: 3.2 Type A intro
# ==============================================================================
md("""### 3.2 Type A 詳細（概念理解）

Type Aは、日本文化の抽象概念を200字以内で説明する課題。4観点（正確性・文化的深度・要点網羅性・簡潔性）で各1-5点の採点。""")

# ==============================================================================
# Cell 10: Fig03 — Type A theme scores
# ==============================================================================
code("""# === Fig 03: Type A Theme Scores (base vs SFT1) ===

focus_models = ['llmjp-v4-8b', 'llm-jp-4-swallow-sft1', 'claude']
df_a = scores_dfs['type_a']

fig, ax = plt.subplots(figsize=(14, 6))
themes_sorted = sorted(THEMES)
x = np.arange(len(themes_sorted))
width = 0.25

for i, m in enumerate(focus_models):
    vals = []
    for t in themes_sorted:
        sub = df_a[(df_a['model'] == m) & (df_a['theme_name'] == t)]
        vals.append(sub['total_score'].mean() if len(sub) > 0 else 0)
    ax.bar(x + i * width - width, vals, width,
           label=MODEL_SHORT[m], color=MODEL_COLORS[m], alpha=0.85)

ax.set_ylabel('平均総合スコア (/20)')
ax.set_title('図3: Type A テーマ別スコア（llm-jp v4 base vs SFT1 vs Claude）')
ax.set_xticks(x)
ax.set_xticklabels(themes_sorted)
ax.legend()
ax.set_ylim(0, 20)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig03_type_a_theme_scores.png')
plt.show()
print("Fig03 saved.")""")

# ==============================================================================
# Cell 11: Fig04 — Type A criterion radar
# ==============================================================================
code("""# === Fig 04: Type A Criterion Balance Radar Chart ===

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
crit_data = {}
for m in focus_models:
    s = summaries['type_a'][m]
    crit_data[m] = [s['overall']['average_per_criterion'][c] for c in CRITERIA_A]

radar_chart(ax, CRITERIA_A, crit_data, '図4: Type A 観点バランス', max_val=5)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig04_type_a_criterion_radar.png')
plt.show()
print("Fig04 saved.")""")

# ==============================================================================
# Cell 12: Fig05 — Type A score distribution
# ==============================================================================
code("""# === Fig 05: Type A Score Distribution Histograms (4 criteria × 1-5) ===

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
df_a = scores_dfs['type_a']

for idx, crit in enumerate(CRITERIA_A):
    ax = axes[idx // 2][idx % 2]
    for m in focus_models:
        sub = df_a[df_a['model'] == m][crit]
        counts = sub.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        ax.bar(np.array([1, 2, 3, 4, 5]) + MODEL_IDS.index(m) * 0.2 - 0.2,
               counts.values, 0.2, label=MODEL_SHORT[m], color=MODEL_COLORS[m], alpha=0.85)
    ax.set_title(f'{crit}')
    ax.set_xlabel('スコア')
    ax.set_ylabel('問題数')
    ax.set_xticks([1, 2, 3, 4, 5])
    if idx == 0:
        ax.legend(fontsize=8)

fig.suptitle('図5: Type A スコア分布（4観点 × 1-5点）', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig05_type_a_score_distribution.png')
plt.show()
print("Fig05 saved.")""")

# ==============================================================================
# Cell 13: 3.3 Type B intro
# ==============================================================================
md("""### 3.3 Type B 詳細（応用力）

Type Bは、抽象概念が具体的な文化要素にどう表れるかを150字以内で説明する課題。4観点（正確性・文化的深度・場面適合性・簡潔性）で採点。""")

# ==============================================================================
# Cell 14: Fig06 + Fig07 — Type B theme and cultural elements
# ==============================================================================
code("""# === Fig 06: Type B Theme Scores (base vs SFT1) ===

df_b = scores_dfs['type_b']

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(themes_sorted))
width = 0.25

for i, m in enumerate(focus_models):
    vals = []
    for t in themes_sorted:
        sub = df_b[(df_b['model'] == m) & (df_b['theme_name'] == t)]
        vals.append(sub['total_score'].mean() if len(sub) > 0 else 0)
    ax.bar(x + i * width - width, vals, width,
           label=MODEL_SHORT[m], color=MODEL_COLORS[m], alpha=0.85)

ax.set_ylabel('平均総合スコア (/20)')
ax.set_title('図6: Type B テーマ別スコア（llm-jp v4 base vs SFT1 vs Claude）')
ax.set_xticks(x)
ax.set_xticklabels(themes_sorted)
ax.legend()
ax.set_ylim(0, 20)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig06_type_b_theme_scores.png')
plt.show()
print("Fig06 saved.")""")

# ==============================================================================
# Cell 15: Fig07 — Cultural elements
# ==============================================================================
code("""# === Fig 07: Type B Cultural Element Scores (Top/Bottom) ===

df_b = scores_dfs['type_b']
# Use llm-jp v4 SFT1 as focus model
m = 'llm-jp-4-swallow-sft1'
elem_scores = df_b[df_b['model'] == m].groupby('cultural_element')['total_score'].mean().sort_values()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Bottom 15
bottom = elem_scores.head(15)
ax1.barh(range(len(bottom)), bottom.values, color='#e74c3c', alpha=0.7)
ax1.set_yticks(range(len(bottom)))
ax1.set_yticklabels(bottom.index)
ax1.set_xlabel('平均スコア (/20)')
ax1.set_title('低得点 文化要素（下位15）')
ax1.set_xlim(0, 20)

# Top 15
top = elem_scores.tail(15)
ax2.barh(range(len(top)), top.values, color='#2196F3', alpha=0.7)
ax2.set_yticks(range(len(top)))
ax2.set_yticklabels(top.index)
ax2.set_xlabel('平均スコア (/20)')
ax2.set_title('高得点 文化要素（上位15）')
ax2.set_xlim(0, 20)

fig.suptitle('図7: Type B 文化要素別スコア（llm-jp v4 SFT1）', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig07_type_b_cultural_elements.png')
plt.show()
print("Fig07 saved.")""")

# ==============================================================================
# Cell 16: Fig08 — Type B theme×criterion heatmap
# ==============================================================================
code("""# === Fig 08: Type B Theme × Criterion Heatmap ===

df_b = scores_dfs['type_b']
m = 'llm-jp-4-swallow-sft1'
sub = df_b[df_b['model'] == m]

heatmap_data = []
for t in themes_sorted:
    row = {}
    t_sub = sub[sub['theme_name'] == t]
    for c in CRITERIA_B:
        row[c] = t_sub[c].mean() if len(t_sub) > 0 else 0
    heatmap_data.append(row)

hm_df = pd.DataFrame(heatmap_data, index=themes_sorted)

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(hm_df, annot=True, fmt='.2f', cmap='YlOrRd', vmin=1, vmax=5,
            ax=ax, cbar_kws={'label': 'スコア (1-5)'})
ax.set_title('図8: Type B テーマ × 観点 ヒートマップ（llm-jp v4 SFT1）')
ax.set_ylabel('テーマ')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig08_type_b_theme_criterion_heatmap.png')
plt.show()
print("Fig08 saved.")""")

# ==============================================================================
# Cell 17: Fig09 — Type B SFT improvement
# ==============================================================================
code("""# === Fig 09: Type B SFT Improvement by Theme (Delta) ===

df_b = scores_dfs['type_b']

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(themes_sorted))
width = 0.25

for i, (base, sft, label) in enumerate(SFT_PAIRS):
    deltas = []
    for t in themes_sorted:
        base_val = df_b[(df_b['model'] == base) & (df_b['theme_name'] == t)]['total_score'].mean()
        sft_val = df_b[(df_b['model'] == sft) & (df_b['theme_name'] == t)]['total_score'].mean()
        deltas.append(sft_val - base_val if not (np.isnan(base_val) or np.isnan(sft_val)) else 0)
    colors = ['#2196F3' if d >= 0 else '#e74c3c' for d in deltas]
    ax.bar(x + i * width - width, deltas, width, label=label,
           color=MODEL_COLORS[sft], alpha=0.85)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('スコア改善幅 (Δ)')
ax.set_title('図9: Type B SFT/RL改善幅（テーマ別）')
ax.set_xticks(x)
ax.set_xticklabels(themes_sorted)
ax.legend()
plt.tight_layout()
plt.savefig(FIGDIR / 'fig09_type_b_sft_improvement.png')
plt.show()
print("Fig09 saved.")""")

# ==============================================================================
# Cell 18: 3.4 Type C intro
# ==============================================================================
md("""### 3.4 Type C 詳細（弁別力）

Type Cは、日本の文化概念と外国（中国・西洋・韓国）の類似概念を比較し、日本文化の特徴を踏まえて違いを説明する課題。200字以内で4観点（正確性・文化的深度・対比の明確さ・簡潔性）で採点。""")

# ==============================================================================
# Cell 19: Fig10 — Type C culture×criterion heatmap
# ==============================================================================
code("""# === Fig 10: Type C Culture × Criterion Heatmap ===

fig, axes = plt.subplots(1, len(MODEL_IDS), figsize=(28, 4), sharey=True)

for idx, m in enumerate(MODEL_IDS):
    s = summaries['type_c'][m]
    data = []
    for culture in CULTURES:
        if culture in s.get('by_culture', {}):
            row = s['by_culture'][culture]['average_per_criterion']
            data.append([row.get(c, 0) for c in CRITERIA_C])
        else:
            data.append([0] * len(CRITERIA_C))
    hm = pd.DataFrame(data, index=CULTURES, columns=CRITERIA_C)
    sns.heatmap(hm, annot=True, fmt='.2f', cmap='YlOrRd', vmin=1, vmax=5,
                ax=axes[idx], cbar=idx == len(MODEL_IDS) - 1,
                cbar_kws={'label': 'スコア'} if idx == len(MODEL_IDS) - 1 else {})
    axes[idx].set_title(MODEL_SHORT[m], fontsize=9)
    if idx > 0:
        axes[idx].set_ylabel('')

fig.suptitle('図10: Type C 文化圏 × 観点 ヒートマップ（全モデル）', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig10_type_c_culture_criterion_heatmap.png')
plt.show()
print("Fig10 saved.")""")

# ==============================================================================
# Cell 20: Fig11 — Type C theme×culture cross
# ==============================================================================
code("""# === Fig 11: Type C Theme × Culture Cross-Tabulation ===

df_c = scores_dfs['type_c']
m = 'llm-jp-4-swallow-sft1'
sub = df_c[df_c['model'] == m]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(themes_sorted))
width = 0.25

for i, culture in enumerate(CULTURES):
    vals = []
    for t in themes_sorted:
        t_sub = sub[(sub['theme_name'] == t) & (sub['foreign_culture'] == culture)]
        vals.append(t_sub['total_score'].mean() if len(t_sub) > 0 else np.nan)
    ax.bar(x + i * width - width, vals, width, label=culture, alpha=0.85)

ax.set_ylabel('平均総合スコア (/20)')
ax.set_title('図11: Type C テーマ × 文化圏（llm-jp v4 SFT1）')
ax.set_xticks(x)
ax.set_xticklabels(themes_sorted)
ax.legend()
ax.set_ylim(0, 20)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig11_type_c_theme_culture_cross.png')
plt.show()
print("Fig11 saved.")""")

# ==============================================================================
# Cell 21: Fig12 — Conciseness anomaly scatter
# ==============================================================================
code("""# === Fig 12: Type C Conciseness Anomaly Scatter ===

# Compare conciseness (簡潔性) vs other criteria average
df_c = scores_dfs['type_c']

fig, ax = plt.subplots(figsize=(10, 8))

for m in MODEL_IDS:
    sub = df_c[df_c['model'] == m]
    other_avg = sub[['正確性', '文化的深度', '対比の明確さ']].mean(axis=1)
    conciseness = sub['簡潔性']
    ax.scatter(other_avg, conciseness, label=MODEL_SHORT[m],
               color=MODEL_COLORS[m], alpha=0.6, s=30)

# Add diagonal
ax.plot([1, 5], [1, 5], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel('他3観点の平均スコア')
ax.set_ylabel('簡潔性スコア')
ax.set_title('図12: Type C 簡潔性 vs 他観点（散布図）')
ax.legend(fontsize=8)
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig12_type_c_conciseness_anomaly.png')
plt.show()

# Statistical analysis: Type C conciseness vs Type A/B
for axis_name in ['type_a', 'type_b', 'type_c']:
    vals = scores_dfs[axis_name]['簡潔性'].values
    print(f"{axis_name} 簡潔性: mean={vals.mean():.2f}, std={vals.std():.2f}")

print("\\nFig12 saved.")""")

# ==============================================================================
# Cell 22: 3.5 JamC-QA
# ==============================================================================
md("""### 3.5 JamC-QA 詳細（知識）

JamC-QAは日本文化に関する4択知識問題（100問）。カテゴリは culture（文化）、custom（風習）、regional_identity（風土）の3つ。""")

# ==============================================================================
# Cell 23: Fig13 — JamCQA category breakdown
# ==============================================================================
code("""# === Fig 13: JamC-QA Category-wise Accuracy (8 models) ===

categories = ['culture', 'custom', 'regional_identity']
cat_display = {'culture': '文化', 'custom': '風習', 'regional_identity': '風土'}

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(MODEL_IDS))
width = 0.25

for i, cat in enumerate(categories):
    vals = []
    for m in MODEL_IDS:
        s = summaries['jamcqa'][m]
        if cat in s['by_category']:
            vals.append(s['by_category'][cat]['accuracy'] * 100)
        else:
            vals.append(0)
    ax.bar(x + i * width - width, vals, width,
           label=cat_display[cat], alpha=0.85)

ax.set_ylabel('正解率 (%)')
ax.set_title('図13: JamC-QA カテゴリ別精度（8モデル）')
ax.set_xticks(x)
ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_IDS], rotation=30, ha='right')
ax.legend()
ax.set_ylim(0, 105)
ax.axhline(y=25, color='gray', linestyle=':', alpha=0.5, label='ランダム (25%)')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig13_jamcqa_category_breakdown.png')
plt.show()
print("Fig13 saved.")""")

# ==============================================================================
# Cell 24: Fig14 — JamCQA SFT improvement
# ==============================================================================
code("""# === Fig 14: JamC-QA SFT Improvement (3 pairs) ===

fig, ax = plt.subplots(figsize=(10, 5))

pair_labels = []
base_vals = []
sft_vals = []
for base, sft, label in SFT_PAIRS:
    pair_labels.append(label)
    base_vals.append(summaries['jamcqa'][base]['overall']['accuracy'] * 100)
    sft_vals.append(summaries['jamcqa'][sft]['overall']['accuracy'] * 100)

x = np.arange(len(pair_labels))
width = 0.35
ax.bar(x - width/2, base_vals, width, label='ベース', color='#90CAF9', alpha=0.85)
ax.bar(x + width/2, sft_vals, width, label='SFT/RL後', color='#1565C0', alpha=0.85)

# Add delta labels
for i in range(len(pair_labels)):
    delta = sft_vals[i] - base_vals[i]
    sign = '+' if delta >= 0 else ''
    ax.text(i, max(base_vals[i], sft_vals[i]) + 1, f'{sign}{delta:.1f}%',
            ha='center', fontsize=10, fontweight='bold',
            color='green' if delta >= 0 else 'red')

ax.set_ylabel('正解率 (%)')
ax.set_title('図14: JamC-QA SFT/RL改善分析（3ペア比較）')
ax.set_xticks(x)
ax.set_xticklabels(pair_labels)
ax.legend()
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig14_jamcqa_sft_improvement.png')
plt.show()
print("Fig14 saved.")""")

# ==============================================================================
# Cell 25: 3.6 Cross-axis
# ==============================================================================
md("""### 3.6 軸横断分析""")

# ==============================================================================
# Cell 26: Fig15 — Cross-axis theme radar
# ==============================================================================
code("""# === Fig 15: Cross-Axis Theme Radar Chart ===
# Show llm-jp v4 SFT1 performance across themes for Type A/B/C

m = 'llm-jp-4-swallow-sft1'
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

axis_theme_data = {}
for axis in ['type_a', 'type_b', 'type_c']:
    s = summaries[axis][m]
    vals = []
    for t in themes_sorted:
        if t in s['by_theme']:
            vals.append(s['by_theme'][t]['average_total'])
        else:
            vals.append(0)
    axis_theme_data[AXIS_SHORT[axis]] = vals

N = len(themes_sorted)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], themes_sorted, fontsize=11)

colors_axis = {'Type A': '#2196F3', 'Type B': '#4CAF50', 'Type C': '#FF9800'}
for name, vals in axis_theme_data.items():
    v = vals + vals[:1]
    ax.plot(angles, v, 'o-', linewidth=2, label=name, color=colors_axis[name])
    ax.fill(angles, v, alpha=0.1, color=colors_axis[name])

ax.set_ylim(0, 20)
ax.set_title('図15: テーマ横断レーダーチャート（llm-jp v4 SFT1）', pad=20, fontsize=13)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(FIGDIR / 'fig15_cross_axis_theme_radar.png')
plt.show()
print("Fig15 saved.")""")

# ==============================================================================
# Cell 27: Fig16 — Theme ranking heatmap
# ==============================================================================
code("""# === Fig 16: Model × Theme Ranking Heatmap ===

# Average across Type A/B/C for each model-theme pair
hm_data = []
for m in MODEL_IDS:
    row = {}
    for t in themes_sorted:
        vals = []
        for axis in ['type_a', 'type_b', 'type_c']:
            s = summaries[axis][m]
            if t in s['by_theme']:
                vals.append(s['by_theme'][t]['average_total'])
        row[t] = np.mean(vals) if vals else 0
    hm_data.append(row)

hm_df = pd.DataFrame(hm_data, index=[MODEL_SHORT[m] for m in MODEL_IDS])

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(hm_df, annot=True, fmt='.1f', cmap='YlOrRd', vmin=5, vmax=20,
            ax=ax, cbar_kws={'label': '平均スコア (/20)'})
ax.set_title('図16: モデル × テーマ ランキング ヒートマップ（Type A/B/C平均）')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig16_theme_ranking_heatmap.png')
plt.show()
print("Fig16 saved.")""")

# ==============================================================================
# Cell 28: Fig17 — SFT improvement correlation
# ==============================================================================
code("""# === Fig 17: SFT Improvement Correlation Across Axes ===

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axis_pairs = [('type_a', 'type_b'), ('type_a', 'type_c'), ('type_b', 'type_c')]

for idx, (ax1_name, ax2_name) in enumerate(axis_pairs):
    ax = axes[idx]
    for base, sft, label in SFT_PAIRS:
        deltas_x, deltas_y = [], []
        for t in themes_sorted:
            # Axis 1
            s1_base = summaries[ax1_name][base]['by_theme'].get(t, {}).get('average_total', 0)
            s1_sft = summaries[ax1_name][sft]['by_theme'].get(t, {}).get('average_total', 0)
            # Axis 2
            s2_base = summaries[ax2_name][base]['by_theme'].get(t, {}).get('average_total', 0)
            s2_sft = summaries[ax2_name][sft]['by_theme'].get(t, {}).get('average_total', 0)
            deltas_x.append(s1_sft - s1_base)
            deltas_y.append(s2_sft - s2_base)
        ax.scatter(deltas_x, deltas_y, label=label, alpha=0.7, s=40, color=MODEL_COLORS[sft])
        # Add correlation
        r, p = stats.spearmanr(deltas_x, deltas_y)
        ax.annotate(f'{label}\\nr={r:.2f}, p={p:.3f}',
                    xy=(0.05, 0.95 - idx * 0.15), xycoords='axes fraction',
                    fontsize=7, verticalalignment='top')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel(f'{AXIS_SHORT[ax1_name]} 改善幅')
    ax.set_ylabel(f'{AXIS_SHORT[ax2_name]} 改善幅')
    ax.set_title(f'{AXIS_SHORT[ax1_name]} vs {AXIS_SHORT[ax2_name]}')
    ax.legend(fontsize=7)

fig.suptitle('図17: SFT改善の軸間相関', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig17_sft_improvement_correlation.png')
plt.show()
print("Fig17 saved.")""")

# ==============================================================================
# Cell 29: Chapter 4 intro
# ==============================================================================
md("""## 第4章: ベンチマークの有効性検証

### 4.1 軸間の独立性""")

# ==============================================================================
# Cell 30: Fig18 — Inter-axis correlation
# ==============================================================================
code("""# === Fig 18: Inter-Axis Correlation Matrix (Spearman) ===

# Build model-level scores for correlation
model_axis_scores = {}
for m in MODEL_IDS:
    scores = []
    for axis in AXES:
        s = summaries[axis][m]
        if axis == 'jamcqa':
            scores.append(s['overall']['accuracy'] * 20)  # Normalize
        else:
            scores.append(s['overall']['average_total'])
    model_axis_scores[m] = scores

score_matrix = pd.DataFrame(model_axis_scores, index=[AXIS_SHORT[a] for a in AXES]).T

fig, ax = plt.subplots(figsize=(8, 7))

# Compute Spearman correlation
corr = score_matrix.corr(method='spearman')
p_values = pd.DataFrame(np.zeros((4, 4)), index=corr.index, columns=corr.columns)
for i, c1 in enumerate(corr.columns):
    for j, c2 in enumerate(corr.columns):
        if i != j:
            r, p = stats.spearmanr(score_matrix[c1], score_matrix[c2])
            p_values.iloc[i, j] = p

# Create annotation with correlation + significance
annot = corr.round(2).astype(str)
for i in range(4):
    for j in range(4):
        if i != j:
            p = p_values.iloc[i, j]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            annot.iloc[i, j] = f'{corr.iloc[i,j]:.2f}{sig}'

sns.heatmap(corr, annot=annot, fmt='', cmap='RdBu_r', vmin=-1, vmax=1,
            ax=ax, cbar_kws={'label': 'Spearman ρ'}, square=True)
ax.set_title('図18: 4軸間相関行列（Spearman相関 + 有意水準）\\n* p<.05, ** p<.01, *** p<.001')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig18_inter_axis_correlation.png')
plt.show()

print("\\nSpearman Correlation Matrix:")
print(corr.to_string())
print("\\np-values:")
print(p_values.to_string())
print("\\nFig18 saved.")""")

# ==============================================================================
# Cell 31: 4.2 Model discrimination
# ==============================================================================
md("""### 4.2 モデル弁別力""")

# ==============================================================================
# Cell 32: Fig19 — Model discrimination boxplots
# ==============================================================================
code("""# === Fig 19: Model Score Distribution Boxplots ===

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, axis in enumerate(['type_a', 'type_b', 'type_c']):
    ax = axes[idx]
    df = scores_dfs[axis]
    box_data = [df[df['model'] == m]['total_score'].values for m in MODEL_IDS]
    bp = ax.boxplot(box_data, labels=[MODEL_SHORT[m] for m in MODEL_IDS],
                    patch_artist=True, showmeans=True)
    for patch, m in zip(bp['boxes'], MODEL_IDS):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.6)
    ax.set_title(AXIS_SHORT[axis])
    ax.set_ylabel('総合スコア (/20)')
    ax.set_ylim(0, 21)
    ax.tick_params(axis='x', rotation=45)

fig.suptitle('図19: モデル別スコア分布箱ひげ図', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig19_model_discrimination_boxplots.png')
plt.show()

# Kruskal-Wallis test
print("\\n=== Kruskal-Wallis検定（モデル間差異） ===")
for axis in ['type_a', 'type_b', 'type_c']:
    df = scores_dfs[axis]
    groups = [df[df['model'] == m]['total_score'].values for m in MODEL_IDS]
    stat, p = stats.kruskal(*groups)
    # Effect size (eta-squared)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta2 = (stat - k + 1) / (n - k)
    print(f"{AXIS_SHORT[axis]}: H={stat:.2f}, p={p:.2e}, η²={eta2:.3f}")

# Ceiling/floor effect
print("\\n=== 天井/床効果分析 ===")
for axis in ['type_a', 'type_b', 'type_c']:
    df = scores_dfs[axis]
    for m in MODEL_IDS:
        sub = df[df['model'] == m]['total_score']
        ceil = (sub >= 19).mean() * 100
        floor = (sub <= 5).mean() * 100
        if ceil > 10 or floor > 10:
            print(f"  {AXIS_SHORT[axis]} {MODEL_SHORT[m]}: ceiling={ceil:.1f}%, floor={floor:.1f}%")

print("\\nFig19 saved.")""")

# ==============================================================================
# Cell 33: 4.3 Criterion validity
# ==============================================================================
md("""### 4.3 評価基準の妥当性""")

# ==============================================================================
# Cell 34: Fig20 — Criterion distributions
# ==============================================================================
code("""# === Fig 20: Criterion Score Distributions (Violin plots) ===

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for row_idx, axis in enumerate(['type_a', 'type_b', 'type_c']):
    df = scores_dfs[axis]
    criteria = CRITERIA_BY_TYPE[axis]
    for col_idx, crit in enumerate(criteria):
        ax = axes[row_idx][col_idx]
        plot_data = []
        labels = []
        for m in MODEL_IDS:
            vals = df[df['model'] == m][crit].values
            plot_data.append(vals)
            labels.append(MODEL_SHORT[m])
        parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(MODEL_COLORS[MODEL_IDS[i]])
            pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(MODEL_IDS) + 1))
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_ylim(0.5, 5.5)
        ax.set_title(f'{AXIS_SHORT[axis]} - {crit}', fontsize=9)
        ax.set_ylabel('スコア')

fig.suptitle('図20: 観点別スコア分布（バイオリンプロット）', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig20_criterion_distributions.png')
plt.show()
print("Fig20 saved.")""")

# ==============================================================================
# Cell 35: Fig21 — Conciseness anomaly detail
# ==============================================================================
code("""# === Fig 21: Type C Conciseness Anomaly Detail ===

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Mean conciseness by type
ax = axes[0]
for m in MODEL_IDS:
    vals = []
    for axis in ['type_a', 'type_b', 'type_c']:
        vals.append(scores_dfs[axis][scores_dfs[axis]['model'] == m]['簡潔性'].mean())
    ax.plot(['Type A', 'Type B', 'Type C'], vals, 'o-',
            label=MODEL_SHORT[m], color=MODEL_COLORS[m], alpha=0.8)
ax.set_ylabel('簡潔性平均スコア')
ax.set_title('簡潔性の推移（Type A→C）')
ax.legend(fontsize=7)
ax.set_ylim(1, 5)

# Panel 2: Type C conciseness by culture
ax = axes[1]
df_c = scores_dfs['type_c']
for m in ['llmjp-v4-8b', 'llm-jp-4-swallow-sft1', 'claude']:
    vals = []
    for c in CULTURES:
        sub = df_c[(df_c['model'] == m) & (df_c['foreign_culture'] == c)]
        vals.append(sub['簡潔性'].mean() if len(sub) > 0 else 0)
    ax.bar(np.arange(len(CULTURES)) + MODEL_IDS.index(m) * 0.2 - 0.2,
           vals, 0.2, label=MODEL_SHORT[m], color=MODEL_COLORS[m], alpha=0.85)
ax.set_xticks(range(len(CULTURES)))
ax.set_xticklabels(CULTURES)
ax.set_ylabel('簡潔性スコア')
ax.set_title('文化圏別 簡潔性（Type C）')
ax.legend(fontsize=8)
ax.set_ylim(0, 5)

# Panel 3: Conciseness vs total score scatter (Type C)
ax = axes[2]
for m in MODEL_IDS:
    sub = df_c[df_c['model'] == m]
    non_concise = sub['total_score'] - sub['簡潔性']
    ax.scatter(sub['簡潔性'], non_concise, s=15, alpha=0.4,
               color=MODEL_COLORS[m], label=MODEL_SHORT[m])
ax.set_xlabel('簡潔性スコア')
ax.set_ylabel('他3観点合計 (/15)')
ax.set_title('簡潔性 vs 他観点（Type C）')
ax.legend(fontsize=6, ncol=2)

fig.suptitle('図21: Type C 簡潔性異常の詳細分析', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig21_conciseness_anomaly_detail.png')
plt.show()

# Mann-Whitney U test: Type A conciseness vs Type C conciseness
print("\\n=== Mann-Whitney U検定: Type A vs Type C 簡潔性 ===")
for m in MODEL_IDS:
    a_vals = scores_dfs['type_a'][scores_dfs['type_a']['model'] == m]['簡潔性']
    c_vals = scores_dfs['type_c'][scores_dfs['type_c']['model'] == m]['簡潔性']
    stat, p = stats.mannwhitneyu(a_vals, c_vals, alternative='two-sided')
    print(f"  {MODEL_SHORT[m]}: U={stat:.0f}, p={p:.4f}, "
          f"A_mean={a_vals.mean():.2f}, C_mean={c_vals.mean():.2f}")

print("\\nFig21 saved.")""")

# ==============================================================================
# Cell 36: 4.4 Problem design
# ==============================================================================
md("""### 4.4 問題設計の適切性""")

# ==============================================================================
# Cell 37: Fig22 — Difficulty distribution
# ==============================================================================
code("""# === Fig 22: Question Difficulty Distribution ===

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for idx, axis in enumerate(['type_a', 'type_b', 'type_c']):
    ax = axes[idx]
    df = scores_dfs[axis]
    # Average score per question across all models = difficulty proxy
    q_difficulty = df.groupby('id')['total_score'].mean()
    ax.hist(q_difficulty, bins=15, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=q_difficulty.mean(), color='red', linestyle='--', label=f'平均={q_difficulty.mean():.1f}')
    ax.set_xlabel('平均スコア（全モデル）')
    ax.set_ylabel('問題数')
    ax.set_title(AXIS_SHORT[axis])
    ax.legend()

# JamC-QA difficulty
ax = axes[3]
q_diff_jam = jamcqa_df.groupby('qid')['is_correct'].mean()
ax.hist(q_diff_jam, bins=10, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(x=q_diff_jam.mean(), color='red', linestyle='--', label=f'平均={q_diff_jam.mean():.2f}')
ax.set_xlabel('正答率（全モデル）')
ax.set_ylabel('問題数')
ax.set_title('JamC-QA')
ax.legend()

fig.suptitle('図22: 問題難易度分布', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig22_difficulty_distribution.png')
plt.show()

# Item discrimination
print("\\n=== 項目弁別力分析 ===")
for axis in ['type_a', 'type_b', 'type_c']:
    df = scores_dfs[axis]
    # For each question, compute correlation between question score and model's total average
    model_avgs = df.groupby('model')['total_score'].mean()
    q_ids = df['id'].unique()
    discriminations = []
    for qid in q_ids:
        q_scores = df[df['id'] == qid].set_index('model')['total_score']
        common = q_scores.index.intersection(model_avgs.index)
        if len(common) >= 3:
            r, _ = stats.pearsonr(q_scores[common], model_avgs[common])
            discriminations.append(r)
    discriminations = np.array(discriminations)
    print(f"{AXIS_SHORT[axis]}: mean_disc={discriminations.mean():.3f}, "
          f"good(>0.3)={np.sum(discriminations > 0.3)}/{len(discriminations)}, "
          f"poor(<0.1)={np.sum(discriminations < 0.1)}/{len(discriminations)}")

print("\\nFig22 saved.")""")

# ==============================================================================
# Cell 38: Chapter 5
# ==============================================================================
md("""## 第5章: 含意

### 5.1 日本文化理解の現状""")

# ==============================================================================
# Cell 39: Fig23 + Fig24 — Concept vs Application, Knowledge vs Understanding
# ==============================================================================
code("""# === Fig 23: Concept Understanding vs Application Scatter ===

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Fig 23: Type A (concept) vs Type B (application)
for m in MODEL_IDS:
    x = summaries['type_a'][m]['overall']['average_total']
    y = summaries['type_b'][m]['overall']['average_total']
    ax1.scatter(x, y, s=100, color=MODEL_COLORS[m], zorder=5)
    ax1.annotate(MODEL_SHORT[m], (x, y), textcoords="offset points",
                 xytext=(5, 5), fontsize=8)

ax1.plot([5, 20], [5, 20], 'k--', alpha=0.3)
ax1.set_xlabel('Type A（概念理解）スコア')
ax1.set_ylabel('Type B（応用力）スコア')
ax1.set_title('図23: 概念理解 vs 応用力')
ax1.set_xlim(5, 20)
ax1.set_ylim(5, 20)

# Paired t-test
a_scores = [summaries['type_a'][m]['overall']['average_total'] for m in MODEL_IDS]
b_scores = [summaries['type_b'][m]['overall']['average_total'] for m in MODEL_IDS]
t_stat, p_val = stats.ttest_rel(a_scores, b_scores)
ax1.text(0.05, 0.95, f'対応t検定: t={t_stat:.2f}, p={p_val:.3f}',
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Fig 24: JamC-QA (knowledge) vs Type A+B+C average (understanding)
for m in MODEL_IDS:
    understanding = np.mean([summaries[a][m]['overall']['average_total']
                             for a in ['type_a', 'type_b', 'type_c']])
    knowledge = summaries['jamcqa'][m]['overall']['accuracy'] * 100
    ax2.scatter(knowledge, understanding, s=100, color=MODEL_COLORS[m], zorder=5)
    ax2.annotate(MODEL_SHORT[m], (knowledge, understanding),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)

ax2.set_xlabel('JamC-QA 正解率 (%)')
ax2.set_ylabel('理解力スコア（Type A/B/C平均）')
ax2.set_title('図24: 知識 vs 理解')

# Correlation
know = [summaries['jamcqa'][m]['overall']['accuracy'] * 100 for m in MODEL_IDS]
understand = [np.mean([summaries[a][m]['overall']['average_total']
                        for a in ['type_a', 'type_b', 'type_c']]) for m in MODEL_IDS]
r, p = stats.pearsonr(know, understand)
r_s, p_s = stats.spearmanr(know, understand)
ax2.text(0.05, 0.95, f'Pearson r={r:.2f}, p={p:.3f}\\nSpearman ρ={r_s:.2f}, p={p_s:.3f}',
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(FIGDIR / 'fig23_concept_vs_application.png')
# Save as both fig23 and fig24
fig.savefig(FIGDIR / 'fig24_knowledge_vs_understanding.png')
plt.show()
print("Fig23 & Fig24 saved.")""")

# ==============================================================================
# Cell 40: 5.2 Architecture
# ==============================================================================
md("""### 5.2 モデルアーキテクチャの影響""")

# ==============================================================================
# Cell 41: Fig25 — JP-specialized vs Multilingual
# ==============================================================================
code("""# === Fig 25: JP-Specialized vs Multilingual Comparison ===

# Group models
jp_models = ['llmjp-v4-8b', 'llm-jp-4-swallow-sft1']
multi_models = ['qwen3-8b', 'qwen3-swallow-8b']
oss_models = ['gpt-oss-20b', 'gpt-oss-swallow-20b']
groups = [
    ('llm-jp系\\n(JP特化)', jp_models, '#2196F3'),
    ('Qwen3系\\n(マルチリンガル)', multi_models, '#FF9800'),
    ('GPT-OSS系\\n(英語主体)', oss_models, '#4CAF50'),
    ('Nemotron\\n(日本語対応)', ['nemotron-nano-9b'], '#9E9E9E'),
    ('Claude\\n(API)', ['claude'], '#e74c3c'),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for idx, axis in enumerate(AXES):
    ax = axes[idx]
    x_pos = 0
    for group_name, members, color in groups:
        vals = []
        for m in members:
            s = summaries[axis][m]
            if axis == 'jamcqa':
                vals.append(s['overall']['accuracy'] * 100)
            else:
                vals.append(s['overall']['average_total'])
        ax.bar(x_pos, np.mean(vals), color=color, alpha=0.85, width=0.6)
        if len(vals) > 1:
            ax.errorbar(x_pos, np.mean(vals), yerr=np.std(vals),
                        color='black', capsize=5, capthick=1)
        x_pos += 1
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_title(AXIS_SHORT[axis])
    if axis == 'jamcqa':
        ax.set_ylabel('正解率 (%)')
    else:
        ax.set_ylabel('スコア (/20)')

fig.suptitle('図25: JP特化 vs マルチリンガル 比較', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig25_jp_vs_multilingual.png')
plt.show()
print("Fig25 saved.")""")

# ==============================================================================
# Cell 42: Fig26 — Swallow RL effect
# ==============================================================================
code("""# === Fig 26: Swallow RL Effect ===

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

for idx, axis in enumerate(AXES):
    ax = axes[idx]
    x = np.arange(len(SFT_PAIRS))
    deltas = []
    for base, sft, label in SFT_PAIRS:
        s_base = summaries[axis][base]
        s_sft = summaries[axis][sft]
        if axis == 'jamcqa':
            d = (s_sft['overall']['accuracy'] - s_base['overall']['accuracy']) * 100
        else:
            d = s_sft['overall']['average_total'] - s_base['overall']['average_total']
        deltas.append(d)
    colors = ['green' if d >= 0 else 'red' for d in deltas]
    ax.bar(x, deltas, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, _, l in SFT_PAIRS], fontsize=8, rotation=20)
    ax.set_title(AXIS_SHORT[axis])
    ax.set_ylabel('改善幅 (Δ)')

    # Add value labels
    for i, d in enumerate(deltas):
        sign = '+' if d >= 0 else ''
        ax.text(i, d + (0.3 if d >= 0 else -0.5), f'{sign}{d:.1f}',
                ha='center', fontsize=9, fontweight='bold')

fig.suptitle('図26: Swallow RL/SFT効果（ベース → RL/SFT後 の改善幅）', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig26_swallow_effect.png')
plt.show()
print("Fig26 saved.")""")

# ==============================================================================
# Cell 43: 5.3 SFT improvement
# ==============================================================================
md("""### 5.3 SFTの改善可能性""")

# ==============================================================================
# Cell 44: Fig27 — SFT criterion improvement
# ==============================================================================
code("""# === Fig 27: SFT Criterion-level Improvement ===

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, axis in enumerate(['type_a', 'type_b', 'type_c']):
    ax = axes[idx]
    criteria = CRITERIA_BY_TYPE[axis]
    x = np.arange(len(criteria))
    width = 0.25

    for i, (base, sft, label) in enumerate(SFT_PAIRS):
        deltas = []
        for c in criteria:
            base_val = summaries[axis][base]['overall']['average_per_criterion'][c]
            sft_val = summaries[axis][sft]['overall']['average_per_criterion'][c]
            deltas.append(sft_val - base_val)
        ax.bar(x + i * width - width, deltas, width, label=label,
               color=MODEL_COLORS[sft], alpha=0.85)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, fontsize=9)
    ax.set_title(AXIS_SHORT[axis])
    ax.set_ylabel('改善幅 (Δ)')
    ax.legend(fontsize=7)

fig.suptitle('図27: 観点別SFT/RL改善分析', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig27_sft_criterion_improvement.png')
plt.show()

# Wilcoxon signed-rank test for SFT improvement
print("\\n=== Wilcoxon符号順位検定: SFT改善 ===")
for axis in ['type_a', 'type_b', 'type_c']:
    df = scores_dfs[axis]
    for base, sft, label in SFT_PAIRS:
        base_scores = df[df['model'] == base].sort_values('id')['total_score'].values
        sft_scores = df[df['model'] == sft].sort_values('id')['total_score'].values
        if len(base_scores) == len(sft_scores) and len(base_scores) > 0:
            try:
                stat, p = stats.wilcoxon(sft_scores - base_scores)
                # Effect size (r = Z / sqrt(N))
                n = len(base_scores)
                z = stats.norm.ppf(1 - p/2)
                r = z / np.sqrt(n)
                print(f"  {AXIS_SHORT[axis]} {label}: W={stat:.0f}, p={p:.4f}, "
                      f"r={r:.3f}, mean_Δ={np.mean(sft_scores - base_scores):.2f}")
            except Exception as e:
                print(f"  {AXIS_SHORT[axis]} {label}: {e}")

print("\\nFig27 saved.")""")

# ==============================================================================
# Cell 45: Chapter 6
# ==============================================================================
md("""## 第6章: 課題と限界

### 6.1 ベンチマーク設計上の課題

1. **LLM-as-a-Judge のバイアス**: 評価者としてClaude Sonnet 4.5を使用しているため、Claudeの回答が不当に高く評価される可能性がある。複数の評価者モデルによるクロスバリデーションが必要。

2. **簡潔性の天井効果**: 特にType A/Bでは簡潔性スコアが天井に張り付く傾向があり、弁別力が低い。制約条件（文字数制限）が緩すぎる可能性がある。

3. **Type C簡潔性の逆転現象**: Type Cでは比較対象の説明が必要なため、簡潔性と内容の充実がトレードオフになり、他のTypeとは異なるパターンを示す。簡潔性の基準設計に改善の余地がある。

4. **韓国比較の少なさ**: Type Cの比較対象は中国42件・西洋51件に対し韓国7件と著しく少なく、韓国との比較分析の信頼性が低い。

5. **JamC-QAのカテゴリ偏り**: regional_identity（風土）カテゴリが2問のみで、統計的に信頼性のある分析が困難。

### 6.2 評価方法の限界

1. **参照回答依存性**: LLM-as-a-Judgeは人間が作成した参照回答との比較に基づくため、参照回答の品質が評価結果を大きく左右する。

2. **文化的深度の測定困難性**: 「文化的深度」は主観的な概念であり、一貫した採点基準の適用が難しい。スコアのばらつきが大きい。

3. **再現性の課題**: LLMの確率的な性質上、同一問題への回答やLLM-as-a-Judgeの採点結果が毎回同一とは限らない。

### 6.3 モデル評価の限界

1. **APIモデルとオープンモデルの条件差**: Claudeは非公開のAPIモデルであり、パラメータ数やアーキテクチャが異なるため、直接的な比較には注意が必要。

2. **推論時の条件統一**: 各モデルの推論パラメータ（温度、max_tokens等）が完全に統一されていない可能性がある。""")

# ==============================================================================
# Cell 46: Chapter 7
# ==============================================================================
md("""## 第7章: まとめ

### 主要な知見

1. **Claudeが全軸で最高性能**: Claude Sonnet 4.5は4軸すべてで他モデルを大幅に上回り、日本文化理解のベースラインとして機能する。ただし、LLM-as-a-Judgeバイアスの可能性に留意が必要。

2. **llm-jp v4のSFT効果**: llm-jp v4 → SFT1へのLoRA SFTにより、全軸で改善が見られた。特にType B（応用力）での改善が顕著。

3. **Swallow RLの効果**: Qwen3 → Qwen3-Swallow、GPT-OSS → GPT-OSS-Swallowのいずれも、Swallow RLにより改善が確認された。改善幅はベースモデルの初期性能に依存する傾向がある。

4. **概念理解と応用力のギャップ**: 多くのモデルでType A（概念理解）よりType B（応用力）の方が高スコアとなる傾向があり、具体的な文脈が与えられることで説明能力が向上することを示唆。

5. **文化的深度の共通課題**: 全モデル・全軸で「文化的深度」が最も低いスコアとなる傾向があり、歴史的・精神的背景の深い理解がLLMにとって困難な課題であることを示す。

6. **簡潔性の設計課題**: Type A/Bでは簡潔性が天井効果を示す一方、Type Cでは比較説明の必要性から簡潔性が低下する。評価基準の再設計が課題。

### 今後の方向性

- 複数LLM評価者によるバイアス検証
- 韓国・東南アジアなど比較文化圏の拡充
- 文化的深度の改善に特化したSFT/RLデータセットの構築
- 動的な難易度調整メカニズムの導入
- 多言語展開（英語話者の日本文化理解測定）""")

# ==============================================================================
# Build notebook
# ==============================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

os.makedirs('docs', exist_ok=True)
with open('docs/analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook generated with {len(cells)} cells")
print(f"  Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
