import json
from pathlib import Path
from collections import defaultdict
import numpy as np

eval_dir = Path("results/phase6_blind_evaluation/evaluation")
files = sorted([f for f in eval_dir.glob("*_test_eval.json")])
print(f"Total test eval files found: {len(files)}")

records = []
for f in files:
    with open(f) as fp:
        data = json.load(fp)
    data["filename"] = f.name
    stem = f.name.replace("_test_eval.json", "")
    parts = stem.split("-")
    seed = parts[-1]
    condition = "-".join(parts[2:-1])
    data["seed"] = seed
    data["condition"] = condition
    records.append(data)

# Print per-run details
print("\n" + "=" * 120)
print(f"{'Experiment Filename':<40} | {'Seed':<5} | {'Acc (%)':<8} | {'Sens (%)':<8} | {'Spec (%)':<8} | {'Macro F1':<8} | {'FN':<4} | {'FP':<4}")
print("=" * 120)
for r in records:
    fn = r.get("false_negatives", 0)
    fp = r.get("false_positives", 0)
    print(f"{r['filename']:<40} | {r['seed']:<5} | {r['clean_test_accuracy']:<8.2f} | {r['pneumonia_recall']*100:<8.2f} | {r['normal_recall']*100:<8.2f} | {r['macro_f1']:<8.4f} | {fn:<4} | {fp:<4}")

# Group by condition
grouped = defaultdict(list)
for r in records:
    grouped[r["condition"]].append(r)

print("\n" + "=" * 120)
print(f"{'Condition':<25} | {'Mean Acc (%)':<16} | {'Mean Sens (%)':<16} | {'Mean Spec (%)':<16} | {'Mean Macro F1':<16}")
print("=" * 120)
for cond in sorted(grouped.keys()):
    recs = grouped[cond]
    accs = [r["clean_test_accuracy"] for r in recs]
    sens = [r["pneumonia_recall"] * 100 for r in recs]
    specs = [r["normal_recall"] * 100 for r in recs]
    f1s = [r["macro_f1"] for r in recs]
    print(f"{cond:<25} | {np.mean(accs):6.2f} +/- {np.std(accs):4.2f} | {np.mean(sens):6.2f} +/- {np.std(sens):4.2f} | {np.mean(specs):6.2f} +/- {np.std(specs):4.2f} | {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
print("=" * 120)
