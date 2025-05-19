import json
import torch
import pandas as pd
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_USED = "clip"

# Set result directory
RESULTS_DIR = "base_rev"

# File paths
INPUT_JSON = os.path.join(MODEL_USED, RESULTS_DIR, "base-benchmark-athletes-detailed_results.json")
ATHLETE_METADATA = "benchmark-data-v1/athletes.txt"
OUTPUT_CSV = os.path.join(MODEL_USED, RESULTS_DIR, "base-benchmark-athletes-mean.csv")

# Load model output JSON
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    model_outputs = json.load(f)

# Sport to index mapping
SPORT_INDEX = {
    "Football": 16, "Tennis": 17, "Basketball": 18, "Soccer": 19, "Racing": 20,
    "Martial Arts": 21, "Snowboarding": 22, "Softball": 23, "Baseball": 24,
    "Track": 25, "Golf": 26, "Hockey": 27, "Swimming": 28, "Boxing": 29,
    "Biking": 30, "Gymnastics": 31
}

# Load athlete metadata
athlete_info = pd.read_csv(ATHLETE_METADATA, delimiter='\t')

results = []

# Iterate over models
for model_name, model_data in model_outputs.items():
    selected_logits = []

    # Iterate over images (and corresponding sport info)
    for i, (image_name, logits_dict) in enumerate(model_data.items()):
        sport = athlete_info.loc[i, 'sport']
        target_indices = {0, 9, SPORT_INDEX[sport]}

        logits = [
            logit for j, (_, logit) in enumerate(logits_dict.items()) if j in target_indices
        ]
        selected_logits.append(logits)

    # Compute mean logits
    mean_logits = torch.tensor(selected_logits, dtype=torch.float16).mean(dim=0).to(torch.float32).tolist()

    results.append([
        model_name,
        mean_logits[0],   # "person"
        mean_logits[1],   # "athlete"
        mean_logits[2]    # "specific_sport_mean"
    ])

# Save to CSV
df = pd.DataFrame(results, columns=["model", "person", "athlete", "specific_sport_mean"])
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"Saved CSV file to {OUTPUT_CSV}")
