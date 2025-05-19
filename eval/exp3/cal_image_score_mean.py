import json
import torch
import pandas as pd
import os

MODEL_USED = "clip"

# Set result directory
RESULTS_DIR = "pks_res"

OUTPUT_CSV = os.path.join(MODEL_USED, RESULTS_DIR, "all-image-score-mean.csv")
BRANCHES = ["house", "mayors", "senate", "actors", "athletes"]

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# === Data Collection ===
csv_rows = []

for branch in BRANCHES:
    input_path = os.path.join(MODEL_USED, RESULTS_DIR, f"pks_benchmark_{branch}_detailed_results.json")
    
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    for model_name, individuals_scores in results.items():  # for each model
        matching_scores = [
            score_dict[prompt]
            for idx, (name, score_dict) in enumerate(individuals_scores.items())
            for prompt_idx, prompt in enumerate(score_dict)
            if idx == prompt_idx
        ]
        
        if matching_scores:  # avoid empty list error
            mean_score = torch.tensor(matching_scores, dtype=torch.float16).mean().to(torch.float32).item()
        else:
            mean_score = float('nan')
        
        csv_rows.append([branch, model_name, mean_score])

# === Save CSV ===
df = pd.DataFrame(csv_rows, columns=["file", "model", "image_score"])
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8", float_format="%.6f")

print(f"Saved CSV: {OUTPUT_CSV}")
