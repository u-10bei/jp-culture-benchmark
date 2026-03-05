#!/usr/bin/env python3
"""JamC-QAデータセットのダウンロードとJSONL変換。"""

import json
import os

def main():
    from datasets import load_dataset

    out_dir = os.path.join(os.path.dirname(__file__), "..", "questions", "jamcqa")
    os.makedirs(out_dir, exist_ok=True)

    print("JamC-QAデータセットをダウンロード中...")
    ds = load_dataset("sbintuitions/JamC-QA", "v1.0")

    for split_name, split_data in ds.items():
        out_path = os.path.join(out_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in split_data:
                f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_data)}問 → {out_path}")

    # test_all.jsonl として test をコピー（選定前の全問保存用）
    test_path = os.path.join(out_dir, "test.jsonl")
    test_all_path = os.path.join(out_dir, "test_all.jsonl")
    if os.path.exists(test_path):
        import shutil
        shutil.copy2(test_path, test_all_path)
        print(f"  全テスト問題を test_all.jsonl にコピー")

    # カテゴリ別集計
    with open(test_all_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    categories = {}
    for r in records:
        cat = r.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nカテゴリ別問題数:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}問")

    print(f"\n合計: {len(records)}問")


if __name__ == "__main__":
    main()
