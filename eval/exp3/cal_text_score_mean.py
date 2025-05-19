import json
import torch
import pandas as pd
import os

MODEL_USED = "clip"

# Set result directory
RESULTS_DIR = "pks_rev_res"

OUTPUT_CSV = os.path.join(MODEL_USED, RESULTS_DIR, "all-text-score-mean.csv")

BRANCHES = ["house", "mayors", "senate", "actors", "athletes"]

# Set name of models
MODELS = ["RN50", "RN50x4", "RN50x16", "RN50x64", "ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14@336px"]

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# === Data Collection ===
csv_rows = []

for branch in BRANCHES:
    for model in MODELS:
        input_path = os.path.join(
            MODEL_USED, RESULTS_DIR, f"pks_benchmark_rev_{model}_{branch}_detailed_results.json"
        )

        if not os.path.isfile(input_path):
            print(f"[WARNING] Missing file: {input_path}")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        matching_scores = [
            score_dict[prompt]
            for idx, (name, score_dict) in enumerate(results.items())
            for prompt_idx, prompt in enumerate(score_dict)
            if idx == prompt_idx
        ]

        if matching_scores:
            mean_score = torch.tensor(matching_scores, dtype=torch.float16).mean().to(torch.float32).item()
        else:
            mean_score = float('nan')

        csv_rows.append([branch, model, mean_score])

# === Save CSV ===
df = pd.DataFrame(csv_rows, columns=["file", "model", "text_score"])
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", float_format="%.6f")

print(f"Saved CSV: {OUTPUT_CSV}")
