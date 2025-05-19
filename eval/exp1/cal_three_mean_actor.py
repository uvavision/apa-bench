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
INPUT_JSON = os.path.join(MODEL_USED, RESULTS_DIR, "base-benchmark-actors-detailed_results.json")
ACTOR_METADATA = "benchmark-data-v1/actors.txt"
PROMPT_CSV = "base-prompts-actors.csv"
OUTPUT_CSV = os.path.join(MODEL_USED, RESULTS_DIR, "base-benchmark-actors-mean.csv")

# Load model output JSON
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    model_outputs = json.load(f)

# Load actor metadata (gender, award, etc.)
actor_info = pd.read_csv(ACTOR_METADATA, delimiter='\t')

# Load prompt information (not used here but loaded for completeness)
prompts = pd.read_csv(PROMPT_CSV, delimiter='|')

results = []

# Iterate over models
for model_name, model_data in model_outputs.items():
    selected_logits = []

    # Iterate over images
    for i, (image_name, logits_dict) in enumerate(model_data.items()):
        gender = actor_info.loc[i, 'gender']
        has_award = int(actor_info.loc[i, 'academy_award'])

        logits = []

        for j, (_, logit_value) in enumerate(logits_dict.items()):
            # Select relevant logits based on gender and award
            if gender == "woman" and j in {0, 15, 17 + 2 * has_award}:
                logits.append(logit_value)
            elif gender == "man" and j in {0, 14, 16 + 2 * has_award}:
                logits.append(logit_value)

        selected_logits.append(logits)

    # Compute mean logits across all samples
    mean_logits = torch.tensor(selected_logits, dtype=torch.float16).mean(dim=0).to(torch.float32).tolist()

    results.append([
        model_name,
        mean_logits[0],
        mean_logits[1],
        mean_logits[2]
    ])

# Save results as CSV
df = pd.DataFrame(results, columns=["model", "easy", "middle", "hard"])
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"Saved CSV file to {OUTPUT_CSV}")