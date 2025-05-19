import json
import torch
import pandas as pd
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_USED = "clip"

# Set result directory
RESULTS_DIR = "base_rev"

# Prompt indices for each branch
PROMPT_INDICES = {
    "house": {0, 7, 21},
    "mayors": {0, 7, 19},
    "senate": {0, 7, 17}
}

# Load dummy prompt to get delimiter format (if needed)
_ = pd.read_csv("base-prompts-house.csv", delimiter='|')

# Main loop for each benchmark branch
for branch in ['house', 'mayors', 'senate']:
    input_json_path = os.path.join(MODEL_USED, RESULTS_DIR, f"base-benchmark-{branch}-detailed_results.json")
    target_txt_path = os.path.join("benchmark-data-v1", f"{branch}.txt")
    output_csv_path = os.path.join(MODEL_USED, RESULTS_DIR, f"base-benchmark-{branch}-mean.csv")

    # Load detailed results
    with open(input_json_path, 'r', encoding='utf-8') as f:
        model_results = json.load(f)

    # Load metadata (e.g., per-person info)
    benchmark_info = pd.read_csv(target_txt_path, delimiter='\t')

    aggregated_data = []

    for model_name, model_output in model_results.items():
        selected_logits = []

        # Iterate over each image's logits and metadata row
        for (_, logits_dict), (_, row) in zip(model_output.items(), benchmark_info.iterrows()):
            logits = [
                logit for i, (_, logit) in enumerate(logits_dict.items())
                if i in PROMPT_INDICES[branch]
            ]
            selected_logits.append(logits)

        # Compute mean of selected logits
        mean_logits = torch.tensor(selected_logits, dtype=torch.float16).mean(dim=0).to(torch.float32).tolist()

        aggregated_data.append([
            model_name,
            mean_logits[0],  # "easy"
            mean_logits[1],  # "middle"
            mean_logits[2]   # "hard"
        ])

    # Save to CSV
    df = pd.DataFrame(aggregated_data, columns=["model", "easy", "middle", "hard"])
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Saved CSV file to {output_csv_path}")
