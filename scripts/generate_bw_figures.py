#!/usr/bin/env python3
"""Generate black-and-white versions of all 27 figures for poster printing."""
import json
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

PROJECT_ROOT = Path('/home/llm-user/jp_culture_benchmark')
os.chdir(PROJECT_ROOT)

# Japanese font
_font_path = '/home/llm-user/miniconda3/lib/python3.13/site-packages/japanize_matplotlib/fonts/ipaexg.ttf'
if os.path.exists(_font_path):
    fm.fontManager.addfont(_font_path)
    matplotlib.rcParams['font.family'] = 'IPAexGothic'

# ---------- B&W Style ----------
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'IPAexGothic'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize'] = (10, 6)

FIGDIR = PROJECT_ROOT / 'docs' / 'figures_bw'
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------- B&W Color/Pattern Definitions ----------
# Grayscale values with hatching patterns for bar charts
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

# B&W: grayscale + hatching
MODEL_BW = {
    'claude':                  {'color': '#000000', 'hatch': '',    'marker': 'o', 'ls': '-'},
    'llmjp-v4-8b':             {'color': '#444444', 'hatch': '//',  'marker': 's', 'ls': '--'},
    'llm-jp-4-swallow-sft1':   {'color': '#333333', 'hatch': '\\\\','marker': 'D', 'ls': '-.'},
    'qwen3-8b':                {'color': '#888888', 'hatch': 'xx',  'marker': '^', 'ls': ':'},
    'qwen3-swallow-8b':        {'color': '#777777', 'hatch': '..', 'marker': 'v', 'ls': '-'},
    'gpt-oss-20b':             {'color': '#bbbbbb', 'hatch': '--',  'marker': 'P', 'ls': '--'},
    'gpt-oss-swallow-20b':     {'color': '#999999', 'hatch': '++',  'marker': 'X', 'ls': '-.'},
    'nemotron-nano-9b':        {'color': '#cccccc', 'hatch': 'OO',  'marker': 'h', 'ls': ':'},
}

SFT_PAIRS = [
    ('llmjp-v4-8b', 'llm-jp-4-swallow-sft1', 'llm-jp v4 → SFT1'),
    ('qwen3-8b', 'qwen3-swallow-8b', 'Qwen3 → Swallow'),
    ('gpt-oss-20b', 'gpt-oss-swallow-20b', 'GPT-OSS → Swallow'),
]
SFT_BW = [
    {'color': '#333333', 'hatch': '//', 'hatch2': '\\\\'},
    {'color': '#777777', 'hatch': 'xx', 'hatch2': '..'},
    {'color': '#bbbbbb', 'hatch': '--', 'hatch2': '++'},
]

AXES = ['type_a', 'type_b', 'type_c', 'jamcqa']
AXIS_SHORT = {'type_a': 'Type A', 'type_b': 'Type B', 'type_c': 'Type C', 'jamcqa': 'JamC-QA'}
AXIS_HATCHES = ['', '//', '\\\\', 'xx']

CRITERIA_A = ['正確性', '文化的深度', '要点網羅性', '簡潔性']
CRITERIA_B = ['正確性', '文化的深度', '場面適合性', '簡潔性']
CRITERIA_C = ['正確性', '文化的深度', '対比の明確さ', '簡潔性']
CRITERIA_BY_TYPE = {'type_a': CRITERIA_A, 'type_b': CRITERIA_B, 'type_c': CRITERIA_C}
CULTURES = ['中国', '西洋', '韓国']
THEMES = ['和', '型', '道', '気', '節', '情', '忠', '神', '仏', '縁', '信', '徳', '美']
themes_sorted = sorted(THEMES)

# ---------- Data Loading ----------
def load_summary(axis, model):
    with open(f'results/{axis}/summary_{model}.json') as f:
        return json.load(f)

def load_scores(axis, model):
    rows = []
    with open(f'results/{axis}/scores_{model}.jsonl') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def load_jamcqa_answers(model):
    rows = []
    with open(f'results/jamcqa/answers_{model}.jsonl') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

summaries = {}
for axis in AXES:
    summaries[axis] = {}
    for m in MODEL_IDS:
        summaries[axis][m] = load_summary(axis, m)

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

jamcqa_frames = []
for m in MODEL_IDS:
    for r in load_jamcqa_answers(m):
        r['model'] = m
        jamcqa_frames.append(r)
jamcqa_df = pd.DataFrame(jamcqa_frames)

print("Data loaded.")

# ---------- Helper: B&W radar ----------
def radar_bw(ax, labels, values_dict, title, max_val=None):
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.xticks(angles[:-1], labels, fontsize=10)
    for name, vals in values_dict.items():
        v = vals + vals[:1]
        bw = MODEL_BW[name]
        ax.plot(angles, v, linestyle=bw['ls'], marker=bw['marker'], markersize=5,
                linewidth=2, label=MODEL_SHORT.get(name, name), color=bw['color'])
        ax.fill(angles, v, alpha=0.05, color='gray')
    if max_val:
        ax.set_ylim(0, max_val)
    ax.set_title(title, pad=20, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

focus_models = ['llmjp-v4-8b', 'llm-jp-4-swallow-sft1', 'claude']
FOCUS_GRAYS = {'llmjp-v4-8b': '#aaaaaa', 'llm-jp-4-swallow-sft1': '#555555', 'claude': '#000000'}
FOCUS_HATCHES = {'llmjp-v4-8b': '//', 'llm-jp-4-swallow-sft1': '\\\\', 'claude': ''}

# ============================================================
# Fig 01: Overall Comparison — Bar (Type A-C) + Line (JamC-QA, 2nd axis)
# ============================================================
fig, ax1 = plt.subplots(figsize=(14, 7))
x = np.arange(len(MODEL_IDS))
width = 0.25
bar_axes = ['type_a', 'type_b', 'type_c']
bar_grays = ['#222222', '#888888', '#cccccc']
bar_hatches = ['', '//', '\\\\']

for i, axis in enumerate(bar_axes):
    vals = [summaries[axis][m]['overall']['average_total'] for m in MODEL_IDS]
    ax1.bar(x + i * width - width, vals, width,
            label=AXIS_SHORT[axis], color=bar_grays[i], hatch=bar_hatches[i],
            edgecolor='black', linewidth=0.5)

ax1.set_ylabel('スコア（Type A-C, /20点満点）')
ax1.set_ylim(0, 22)
ax1.axhline(y=20, color='gray', linestyle='--', alpha=0.2)
ax1.set_xticks(x)
ax1.set_xticklabels([MODEL_SHORT[m] for m in MODEL_IDS], rotation=30, ha='right')

# Secondary y-axis: JamC-QA as line
ax2 = ax1.twinx()
jamcqa_vals = [summaries['jamcqa'][m]['overall']['accuracy'] * 100 for m in MODEL_IDS]
ax2.plot(x, jamcqa_vals, color='black', marker='D', markersize=8,
         linewidth=2.5, linestyle='--', label='JamC-QA（精度%）', zorder=5)
ax2.set_ylabel('JamC-QA 精度（%）')
ax2.set_ylim(0, 105)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

ax1.set_title('図1: 8モデル × 4軸 総合比較')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig01_overall_comparison.png')
plt.close()
print("Fig01")

# ============================================================
# Fig 02: Radar chart
# ============================================================
radar_models = ['llmjp-v4-8b', 'llm-jp-4-swallow-sft1', 'claude']
radar_labels = ['Type A\n(概念理解)', 'Type B\n(応用力)', 'Type C\n(弁別力)', 'JamC-QA\n(知識)']
radar_data = {}
for m in radar_models:
    vals = []
    for axis in AXES:
        s = summaries[axis][m]
        vals.append(s['overall']['accuracy'] * 20 if axis == 'jamcqa' else s['overall']['average_total'])
    radar_data[m] = vals

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
radar_bw(ax, radar_labels, radar_data, '図2: llm-jp v4 / SFT1 / Claude\n4軸レーダーチャート', max_val=20)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig02_llmjp_radar_4axis.png')
plt.close()
print("Fig02")

# ============================================================
# Fig 03: Type A theme scores
# ============================================================
df_a = scores_dfs['type_a']
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(themes_sorted))
width = 0.25
for i, m in enumerate(focus_models):
    vals = [df_a[(df_a['model'] == m) & (df_a['theme_name'] == t)]['total_score'].mean() for t in themes_sorted]
    ax.bar(x + i * width - width, vals, width, label=MODEL_SHORT[m],
           color=FOCUS_GRAYS[m], hatch=FOCUS_HATCHES[m], edgecolor='black', linewidth=0.5)
ax.set_ylabel('平均総合スコア (/20)')
ax.set_title('図3: Type A テーマ別スコア（llm-jp v4 base vs SFT1 vs Claude）')
ax.set_xticks(x)
ax.set_xticklabels(themes_sorted)
ax.legend()
ax.set_ylim(0, 20)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig03_type_a_theme_scores.png')
plt.close()
print("Fig03")

# ============================================================
# Fig 04: Type A criterion radar
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
crit_data = {m: [summaries['type_a'][m]['overall']['average_per_criterion'][c] for c in CRITERIA_A] for m in focus_models}
radar_bw(ax, CRITERIA_A, crit_data, '図4: Type A 観点バランス', max_val=5)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig04_type_a_criterion_radar.png')
plt.close()
print("Fig04")

# ============================================================
# Fig 05: Type A score distribution
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, crit in enumerate(CRITERIA_A):
    ax = axes[idx // 2][idx % 2]
    for i, m in enumerate(focus_models):
        sub = df_a[df_a['model'] == m][crit]
        counts = sub.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        ax.bar(np.array([1, 2, 3, 4, 5]) + i * 0.25 - 0.25, counts.values, 0.25,
               label=MODEL_SHORT[m], color=FOCUS_GRAYS[m], hatch=FOCUS_HATCHES[m],
               edgecolor='black', linewidth=0.5)
    ax.set_title(crit)
    ax.set_xlabel('スコア')
    ax.set_ylabel('問題数')
    ax.set_xticks([1, 2, 3, 4, 5])
    if idx == 0:
        ax.legend(fontsize=8)
fig.suptitle('図5: Type A スコア分布（4観点 × 1-5点）', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig05_type_a_score_distribution.png')
plt.close()
print("Fig05")

# ============================================================
# Fig 06: Type B theme scores
# ============================================================
df_b = scores_dfs['type_b']
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(themes_sorted))
width = 0.25
for i, m in enumerate(focus_models):
    vals = [df_b[(df_b['model'] == m) & (df_b['theme_name'] == t)]['total_score'].mean() for t in themes_sorted]
    ax.bar(x + i * width - width, vals, width, label=MODEL_SHORT[m],
           color=FOCUS_GRAYS[m], hatch=FOCUS_HATCHES[m], edgecolor='black', linewidth=0.5)
ax.set_ylabel('平均総合スコア (/20)')
ax.set_title('図6: Type B テーマ別スコア（llm-jp v4 base vs SFT1 vs Claude）')
ax.set_xticks(x)
ax.set_xticklabels(themes_sorted)
ax.legend()
ax.set_ylim(0, 20)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig06_type_b_theme_scores.png')
plt.close()
print("Fig06")

# ============================================================
# Fig 07: Type B cultural elements
# ============================================================
m = 'llm-jp-4-swallow-sft1'
elem_scores = df_b[df_b['model'] == m].groupby('cultural_element')['total_score'].mean().sort_values()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
bottom = elem_scores.head(15)
ax1.barh(range(len(bottom)), bottom.values, color='#aaaaaa', hatch='//', edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(bottom)))
ax1.set_yticklabels(bottom.index)
ax1.set_xlabel('平均スコア (/20)')
ax1.set_title('低得点 文化要素（下位15）')
ax1.set_xlim(0, 20)
top = elem_scores.tail(15)
ax2.barh(range(len(top)), top.values, color='#555555', hatch='\\\\', edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(top)))
ax2.set_yticklabels(top.index)
ax2.set_xlabel('平均スコア (/20)')
ax2.set_title('高得点 文化要素（上位15）')
ax2.set_xlim(0, 20)
fig.suptitle('図7: Type B 文化要素別スコア（llm-jp v4 SFT1）', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig07_type_b_cultural_elements.png')
plt.close()
print("Fig07")

# ============================================================
# Fig 08: Type B theme×criterion heatmap
# ============================================================
m = 'llm-jp-4-swallow-sft1'
sub = df_b[df_b['model'] == m]
hm_data = []
for t in themes_sorted:
    row = {}
    t_sub = sub[sub['theme_name'] == t]
    for c in CRITERIA_B:
        row[c] = t_sub[c].mean() if len(t_sub) > 0 else 0
    hm_data.append(row)
hm_df = pd.DataFrame(hm_data, index=themes_sorted)
fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(hm_df, annot=True, fmt='.2f', cmap='Greys', vmin=1, vmax=5,
            ax=ax, cbar_kws={'label': 'スコア (1-5)'}, linewidths=0.5, linecolor='black')
ax.set_title('図8: Type B テーマ × 観点 ヒートマップ（llm-jp v4 SFT1）')
ax.set_ylabel('テーマ')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig08_type_b_theme_criterion_heatmap.png')
plt.close()
print("Fig08")

# ============================================================
# Fig 09: Type B SFT improvement
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(themes_sorted))
width = 0.25
for i, (base, sft, label) in enumerate(SFT_PAIRS):
    deltas = []
    for t in themes_sorted:
        bv = df_b[(df_b['model'] == base) & (df_b['theme_name'] == t)]['total_score'].mean()
        sv = df_b[(df_b['model'] == sft) & (df_b['theme_name'] == t)]['total_score'].mean()
        deltas.append(sv - bv if not (np.isnan(bv) or np.isnan(sv)) else 0)
    ax.bar(x + i * width - width, deltas, width, label=label,
           color=SFT_BW[i]['color'], hatch=SFT_BW[i]['hatch'], edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('スコア改善幅 (Δ)')
ax.set_title('図9: Type B SFT/RL改善幅（テーマ別）')
ax.set_xticks(x)
ax.set_xticklabels(themes_sorted)
ax.legend()
plt.tight_layout()
plt.savefig(FIGDIR / 'fig09_type_b_sft_improvement.png')
plt.close()
print("Fig09")

# ============================================================
# Fig 10: Type C culture×criterion heatmap
# ============================================================
fig, axes_arr = plt.subplots(1, len(MODEL_IDS), figsize=(28, 4), sharey=True)
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
    sns.heatmap(hm, annot=True, fmt='.2f', cmap='Greys', vmin=1, vmax=5,
                ax=axes_arr[idx], cbar=idx == len(MODEL_IDS) - 1,
                cbar_kws={'label': 'スコア'} if idx == len(MODEL_IDS) - 1 else {},
                linewidths=0.5, linecolor='black')
    axes_arr[idx].set_title(MODEL_SHORT[m], fontsize=9)
    if idx > 0:
        axes_arr[idx].set_ylabel('')
fig.suptitle('図10: Type C 文化圏 × 観点 ヒートマップ（全モデル）', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig10_type_c_culture_criterion_heatmap.png')
plt.close()
print("Fig10")

# ============================================================
# Fig 11: Type C theme×culture cross
# ============================================================
df_c = scores_dfs['type_c']
m = 'llm-jp-4-swallow-sft1'
sub = df_c[df_c['model'] == m]
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(themes_sorted))
width = 0.25
culture_grays = ['#333333', '#888888', '#cccccc']
culture_hatches = ['', '//', 'xx']
for i, culture in enumerate(CULTURES):
    vals = []
    for t in themes_sorted:
        t_sub = sub[(sub['theme_name'] == t) & (sub['foreign_culture'] == culture)]
        vals.append(t_sub['total_score'].mean() if len(t_sub) > 0 else np.nan)
    ax.bar(x + i * width - width, vals, width, label=culture,
           color=culture_grays[i], hatch=culture_hatches[i], edgecolor='black', linewidth=0.5)
ax.set_ylabel('平均総合スコア (/20)')
ax.set_title('図11: Type C テーマ × 文化圏（llm-jp v4 SFT1）')
ax.set_xticks(x)
ax.set_xticklabels(themes_sorted)
ax.legend()
ax.set_ylim(0, 20)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig11_type_c_theme_culture_cross.png')
plt.close()
print("Fig11")

# ============================================================
# Fig 12: Conciseness anomaly scatter
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
for m in MODEL_IDS:
    sub = df_c[df_c['model'] == m]
    other_avg = sub[['正確性', '文化的深度', '対比の明確さ']].mean(axis=1)
    bw = MODEL_BW[m]
    ax.scatter(other_avg, sub['簡潔性'], label=MODEL_SHORT[m],
               color=bw['color'], marker=bw['marker'], alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
ax.plot([1, 5], [1, 5], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel('他3観点の平均スコア')
ax.set_ylabel('簡潔性スコア')
ax.set_title('図12: Type C 簡潔性 vs 他観点（散布図）')
ax.legend(fontsize=7)
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig12_type_c_conciseness_anomaly.png')
plt.close()
print("Fig12")

# ============================================================
# Fig 13: JamCQA category breakdown
# ============================================================
categories = ['culture', 'custom', 'regional_identity']
cat_display = {'culture': '文化', 'custom': '風習', 'regional_identity': '風土'}
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(MODEL_IDS))
width = 0.25
cat_grays = ['#444444', '#999999', '#dddddd']
cat_hatches = ['', '//', 'xx']
for i, cat in enumerate(categories):
    vals = []
    for m in MODEL_IDS:
        s = summaries['jamcqa'][m]
        vals.append(s['by_category'][cat]['accuracy'] * 100 if cat in s['by_category'] else 0)
    ax.bar(x + i * width - width, vals, width, label=cat_display[cat],
           color=cat_grays[i], hatch=cat_hatches[i], edgecolor='black', linewidth=0.5)
ax.set_ylabel('正解率 (%)')
ax.set_title('図13: JamC-QA カテゴリ別精度（8モデル）')
ax.set_xticks(x)
ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_IDS], rotation=30, ha='right')
ax.legend()
ax.set_ylim(0, 105)
ax.axhline(y=25, color='gray', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig13_jamcqa_category_breakdown.png')
plt.close()
print("Fig13")

# ============================================================
# Fig 14: JamCQA SFT improvement
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
pair_labels = [l for _, _, l in SFT_PAIRS]
base_vals = [summaries['jamcqa'][b]['overall']['accuracy'] * 100 for b, _, _ in SFT_PAIRS]
sft_vals = [summaries['jamcqa'][s]['overall']['accuracy'] * 100 for _, s, _ in SFT_PAIRS]
x = np.arange(len(pair_labels))
width = 0.35
ax.bar(x - width/2, base_vals, width, label='ベース', color='#cccccc', hatch='//', edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, sft_vals, width, label='SFT/RL後', color='#555555', hatch='\\\\', edgecolor='black', linewidth=0.5)
for i in range(len(pair_labels)):
    delta = sft_vals[i] - base_vals[i]
    sign = '+' if delta >= 0 else ''
    ax.text(i, max(base_vals[i], sft_vals[i]) + 1, f'{sign}{delta:.1f}%',
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('正解率 (%)')
ax.set_title('図14: JamC-QA SFT/RL改善分析（3ペア比較）')
ax.set_xticks(x)
ax.set_xticklabels(pair_labels)
ax.legend()
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig14_jamcqa_sft_improvement.png')
plt.close()
print("Fig14")

# ============================================================
# Fig 15: Cross-axis theme radar
# ============================================================
m = 'llm-jp-4-swallow-sft1'
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
N = len(themes_sorted)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], themes_sorted, fontsize=11)
axis_ls = {'Type A': '-', 'Type B': '--', 'Type C': '-.'}
axis_markers = {'Type A': 'o', 'Type B': 's', 'Type C': '^'}
axis_grays = {'Type A': '#000000', 'Type B': '#666666', 'Type C': '#aaaaaa'}
for axis_name in ['type_a', 'type_b', 'type_c']:
    s = summaries[axis_name][m]
    vals = [s['by_theme'].get(t, {}).get('average_total', 0) for t in themes_sorted]
    v = vals + vals[:1]
    label = AXIS_SHORT[axis_name]
    ax.plot(angles, v, linestyle=axis_ls[label], marker=axis_markers[label],
            markersize=5, linewidth=2, label=label, color=axis_grays[label])
    ax.fill(angles, v, alpha=0.05, color='gray')
ax.set_ylim(0, 20)
ax.set_title('図15: テーマ横断レーダーチャート（llm-jp v4 SFT1）', pad=20, fontsize=13)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(FIGDIR / 'fig15_cross_axis_theme_radar.png')
plt.close()
print("Fig15")

# ============================================================
# Fig 16: Theme ranking heatmap
# ============================================================
hm_data = []
for m in MODEL_IDS:
    row = {}
    for t in themes_sorted:
        vals = [summaries[a][m]['by_theme'].get(t, {}).get('average_total', 0) for a in ['type_a', 'type_b', 'type_c']]
        row[t] = np.mean(vals) if vals else 0
    hm_data.append(row)
hm_df = pd.DataFrame(hm_data, index=[MODEL_SHORT[m] for m in MODEL_IDS])
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(hm_df, annot=True, fmt='.1f', cmap='Greys', vmin=5, vmax=20,
            ax=ax, cbar_kws={'label': '平均スコア (/20)'}, linewidths=0.5, linecolor='black')
ax.set_title('図16: モデル × テーマ ランキング ヒートマップ（Type A/B/C平均）')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig16_theme_ranking_heatmap.png')
plt.close()
print("Fig16")

# ============================================================
# Fig 17: SFT improvement correlation
# ============================================================
fig, axes_arr = plt.subplots(1, 3, figsize=(18, 5))
axis_pairs = [('type_a', 'type_b'), ('type_a', 'type_c'), ('type_b', 'type_c')]
for idx, (a1, a2) in enumerate(axis_pairs):
    ax = axes_arr[idx]
    for i, (base, sft, label) in enumerate(SFT_PAIRS):
        dx, dy = [], []
        for t in themes_sorted:
            dx.append(summaries[a1][sft]['by_theme'].get(t, {}).get('average_total', 0) -
                      summaries[a1][base]['by_theme'].get(t, {}).get('average_total', 0))
            dy.append(summaries[a2][sft]['by_theme'].get(t, {}).get('average_total', 0) -
                      summaries[a2][base]['by_theme'].get(t, {}).get('average_total', 0))
        bw = MODEL_BW[sft]
        ax.scatter(dx, dy, label=label, alpha=0.7, s=40, color=bw['color'],
                   marker=bw['marker'], edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel(f'{AXIS_SHORT[a1]} 改善幅')
    ax.set_ylabel(f'{AXIS_SHORT[a2]} 改善幅')
    ax.set_title(f'{AXIS_SHORT[a1]} vs {AXIS_SHORT[a2]}')
    ax.legend(fontsize=7)
fig.suptitle('図17: SFT改善の軸間相関', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig17_sft_improvement_correlation.png')
plt.close()
print("Fig17")

# ============================================================
# Fig 18: Inter-axis correlation
# ============================================================
model_axis_scores = {}
for m in MODEL_IDS:
    scores = []
    for axis in AXES:
        s = summaries[axis][m]
        scores.append(s['overall']['accuracy'] * 20 if axis == 'jamcqa' else s['overall']['average_total'])
    model_axis_scores[m] = scores
score_matrix = pd.DataFrame(model_axis_scores, index=[AXIS_SHORT[a] for a in AXES]).T
corr = score_matrix.corr(method='spearman')
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Greys', vmin=-1, vmax=1,
            ax=ax, cbar_kws={'label': 'Spearman ρ'}, square=True,
            linewidths=0.5, linecolor='black')
ax.set_title('図18: 4軸間相関行列（Spearman相関）')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig18_inter_axis_correlation.png')
plt.close()
print("Fig18")

# ============================================================
# Fig 19: Model discrimination boxplots
# ============================================================
fig, axes_arr = plt.subplots(1, 3, figsize=(18, 6))
for idx, axis in enumerate(['type_a', 'type_b', 'type_c']):
    ax = axes_arr[idx]
    df = scores_dfs[axis]
    box_data = [df[df['model'] == m]['total_score'].values for m in MODEL_IDS]
    bp = ax.boxplot(box_data, labels=[MODEL_SHORT[m] for m in MODEL_IDS],
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black', markersize=5),
                    medianprops=dict(color='black', linewidth=1.5))
    for j, patch in enumerate(bp['boxes']):
        gray = MODEL_BW[MODEL_IDS[j]]['color']
        patch.set_facecolor(gray)
        patch.set_edgecolor('black')
        patch.set_alpha(0.6)
    ax.set_title(AXIS_SHORT[axis])
    ax.set_ylabel('総合スコア (/20)')
    ax.set_ylim(0, 21)
    ax.tick_params(axis='x', rotation=45)
fig.suptitle('図19: モデル別スコア分布箱ひげ図', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig19_model_discrimination_boxplots.png')
plt.close()
print("Fig19")

# ============================================================
# Fig 20: Criterion distributions (violin)
# ============================================================
fig, axes_arr = plt.subplots(3, 4, figsize=(20, 15))
for row_idx, axis in enumerate(['type_a', 'type_b', 'type_c']):
    df = scores_dfs[axis]
    criteria = CRITERIA_BY_TYPE[axis]
    for col_idx, crit in enumerate(criteria):
        ax = axes_arr[row_idx][col_idx]
        plot_data = [df[df['model'] == m][crit].values for m in MODEL_IDS]
        parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(MODEL_BW[MODEL_IDS[i]]['color'])
            pc.set_edgecolor('black')
            pc.set_alpha(0.5)
        ax.set_xticks(range(1, len(MODEL_IDS) + 1))
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_IDS], rotation=45, fontsize=7)
        ax.set_ylim(0.5, 5.5)
        ax.set_title(f'{AXIS_SHORT[axis]} - {crit}', fontsize=9)
        ax.set_ylabel('スコア')
fig.suptitle('図20: 観点別スコア分布（バイオリンプロット）', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig20_criterion_distributions.png')
plt.close()
print("Fig20")

# ============================================================
# Fig 21: Conciseness anomaly detail
# ============================================================
fig, axes_arr = plt.subplots(1, 3, figsize=(18, 5))
# Panel 1
ax = axes_arr[0]
for m in MODEL_IDS:
    vals = [scores_dfs[a][scores_dfs[a]['model'] == m]['簡潔性'].mean() for a in ['type_a', 'type_b', 'type_c']]
    bw = MODEL_BW[m]
    ax.plot(['Type A', 'Type B', 'Type C'], vals, linestyle=bw['ls'], marker=bw['marker'],
            label=MODEL_SHORT[m], color=bw['color'], markersize=5)
ax.set_ylabel('簡潔性平均スコア')
ax.set_title('簡潔性の推移（Type A→C）')
ax.legend(fontsize=6, ncol=2)
ax.set_ylim(1, 5)
# Panel 2
ax = axes_arr[1]
for i, m in enumerate(focus_models):
    vals = []
    for c in CULTURES:
        sub = df_c[(df_c['model'] == m) & (df_c['foreign_culture'] == c)]
        vals.append(sub['簡潔性'].mean() if len(sub) > 0 else 0)
    ax.bar(np.arange(len(CULTURES)) + i * 0.25 - 0.25, vals, 0.25,
           label=MODEL_SHORT[m], color=FOCUS_GRAYS[m], hatch=FOCUS_HATCHES[m],
           edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(CULTURES)))
ax.set_xticklabels(CULTURES)
ax.set_ylabel('簡潔性スコア')
ax.set_title('文化圏別 簡潔性（Type C）')
ax.legend(fontsize=8)
ax.set_ylim(0, 5)
# Panel 3
ax = axes_arr[2]
for m in MODEL_IDS:
    sub = df_c[df_c['model'] == m]
    non_c = sub['total_score'] - sub['簡潔性']
    bw = MODEL_BW[m]
    ax.scatter(sub['簡潔性'], non_c, s=15, alpha=0.4, color=bw['color'],
               marker=bw['marker'], edgecolors='black', linewidth=0.2, label=MODEL_SHORT[m])
ax.set_xlabel('簡潔性スコア')
ax.set_ylabel('他3観点合計 (/15)')
ax.set_title('簡潔性 vs 他観点（Type C）')
ax.legend(fontsize=5, ncol=2)
fig.suptitle('図21: Type C 簡潔性異常の詳細分析', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig21_conciseness_anomaly_detail.png')
plt.close()
print("Fig21")

# ============================================================
# Fig 22: Difficulty distribution
# ============================================================
fig, axes_arr = plt.subplots(1, 4, figsize=(20, 5))
for idx, axis in enumerate(['type_a', 'type_b', 'type_c']):
    ax = axes_arr[idx]
    df = scores_dfs[axis]
    q_diff = df.groupby('id')['total_score'].mean()
    ax.hist(q_diff, bins=15, color='#888888', edgecolor='black', linewidth=0.5)
    ax.axvline(x=q_diff.mean(), color='black', linestyle='--', label=f'平均={q_diff.mean():.1f}')
    ax.set_xlabel('平均スコア（全モデル）')
    ax.set_ylabel('問題数')
    ax.set_title(AXIS_SHORT[axis])
    ax.legend()
ax = axes_arr[3]
q_diff_jam = jamcqa_df.groupby('qid')['is_correct'].mean()
ax.hist(q_diff_jam, bins=10, color='#888888', edgecolor='black', linewidth=0.5)
ax.axvline(x=q_diff_jam.mean(), color='black', linestyle='--', label=f'平均={q_diff_jam.mean():.2f}')
ax.set_xlabel('正答率（全モデル）')
ax.set_ylabel('問題数')
ax.set_title('JamC-QA')
ax.legend()
fig.suptitle('図22: 問題難易度分布', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig22_difficulty_distribution.png')
plt.close()
print("Fig22")

# ============================================================
# Fig 23 + 24: Concept vs Application, Knowledge vs Understanding
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for m in MODEL_IDS:
    xv = summaries['type_a'][m]['overall']['average_total']
    yv = summaries['type_b'][m]['overall']['average_total']
    bw = MODEL_BW[m]
    ax1.scatter(xv, yv, s=100, color=bw['color'], marker=bw['marker'],
                edgecolors='black', linewidth=0.5, zorder=5)
    ax1.annotate(MODEL_SHORT[m], (xv, yv), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax1.plot([5, 20], [5, 20], 'k--', alpha=0.3)
ax1.set_xlabel('Type A（概念理解）スコア')
ax1.set_ylabel('Type B（応用力）スコア')
ax1.set_title('図23: 概念理解 vs 応用力')
ax1.set_xlim(5, 20)
ax1.set_ylim(5, 20)
for m in MODEL_IDS:
    understanding = np.mean([summaries[a][m]['overall']['average_total'] for a in ['type_a', 'type_b', 'type_c']])
    knowledge = summaries['jamcqa'][m]['overall']['accuracy'] * 100
    bw = MODEL_BW[m]
    ax2.scatter(knowledge, understanding, s=100, color=bw['color'], marker=bw['marker'],
                edgecolors='black', linewidth=0.5, zorder=5)
    ax2.annotate(MODEL_SHORT[m], (knowledge, understanding), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax2.set_xlabel('JamC-QA 正解率 (%)')
ax2.set_ylabel('理解力スコア（Type A/B/C平均）')
ax2.set_title('図24: 知識 vs 理解')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig23_concept_vs_application.png')
fig.savefig(FIGDIR / 'fig24_knowledge_vs_understanding.png')
plt.close()
print("Fig23 & Fig24")

# ============================================================
# Fig 25: JP vs multilingual
# ============================================================
groups = [
    ('llm-jp系\n(JP特化)', ['llmjp-v4-8b', 'llm-jp-4-swallow-sft1'], '#555555', '//'),
    ('Qwen3系\n(マルチリンガル)', ['qwen3-8b', 'qwen3-swallow-8b'], '#888888', '\\\\'),
    ('GPT-OSS系\n(英語主体)', ['gpt-oss-20b', 'gpt-oss-swallow-20b'], '#bbbbbb', 'xx'),
    ('Nemotron\n(日本語対応)', ['nemotron-nano-9b'], '#dddddd', '..'),
    ('Claude\n(API)', ['claude'], '#333333', ''),
]
fig, axes_arr = plt.subplots(1, 4, figsize=(20, 5))
for idx, axis in enumerate(AXES):
    ax = axes_arr[idx]
    x_pos = 0
    for gn, members, gc, gh in groups:
        vals = []
        for m in members:
            s = summaries[axis][m]
            vals.append(s['overall']['accuracy'] * 100 if axis == 'jamcqa' else s['overall']['average_total'])
        ax.bar(x_pos, np.mean(vals), color=gc, hatch=gh, edgecolor='black', linewidth=0.5, width=0.6)
        if len(vals) > 1:
            ax.errorbar(x_pos, np.mean(vals), yerr=np.std(vals), color='black', capsize=5, capthick=1)
        x_pos += 1
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_title(AXIS_SHORT[axis])
    ax.set_ylabel('正解率 (%)' if axis == 'jamcqa' else 'スコア (/20)')
fig.suptitle('図25: JP特化 vs マルチリンガル 比較', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig25_jp_vs_multilingual.png')
plt.close()
print("Fig25")

# ============================================================
# Fig 26: Swallow RL effect
# ============================================================
fig, axes_arr = plt.subplots(1, 4, figsize=(16, 5))
for idx, axis in enumerate(AXES):
    ax = axes_arr[idx]
    x = np.arange(len(SFT_PAIRS))
    deltas = []
    for base, sft, label in SFT_PAIRS:
        sb = summaries[axis][base]
        ss = summaries[axis][sft]
        d = (ss['overall']['accuracy'] - sb['overall']['accuracy']) * 100 if axis == 'jamcqa' \
            else ss['overall']['average_total'] - sb['overall']['average_total']
        deltas.append(d)
    for i, d in enumerate(deltas):
        ax.bar(i, d, color='#555555' if d >= 0 else '#cccccc',
               hatch=SFT_BW[i]['hatch'], edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, _, l in SFT_PAIRS], fontsize=8, rotation=20)
    ax.set_title(AXIS_SHORT[axis])
    ax.set_ylabel('改善幅 (Δ)')
    for i, d in enumerate(deltas):
        sign = '+' if d >= 0 else ''
        ax.text(i, d + (0.3 if d >= 0 else -0.5), f'{sign}{d:.1f}', ha='center', fontsize=9, fontweight='bold')
fig.suptitle('図26: Swallow RL/SFT効果（ベース → RL/SFT後 の改善幅）', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig26_swallow_effect.png')
plt.close()
print("Fig26")

# ============================================================
# Fig 27: SFT criterion improvement
# ============================================================
fig, axes_arr = plt.subplots(1, 3, figsize=(18, 6))
for idx, axis in enumerate(['type_a', 'type_b', 'type_c']):
    ax = axes_arr[idx]
    criteria = CRITERIA_BY_TYPE[axis]
    x = np.arange(len(criteria))
    width = 0.25
    for i, (base, sft, label) in enumerate(SFT_PAIRS):
        deltas = [summaries[axis][sft]['overall']['average_per_criterion'][c] -
                  summaries[axis][base]['overall']['average_per_criterion'][c] for c in criteria]
        ax.bar(x + i * width - width, deltas, width, label=label,
               color=SFT_BW[i]['color'], hatch=SFT_BW[i]['hatch'], edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, fontsize=9)
    ax.set_title(AXIS_SHORT[axis])
    ax.set_ylabel('改善幅 (Δ)')
    ax.legend(fontsize=7)
fig.suptitle('図27: 観点別SFT/RL改善分析', fontsize=14)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig27_sft_criterion_improvement.png')
plt.close()
print("Fig27")

print(f"\n=== All 27 B&W figures saved to {FIGDIR} ===")
