import json
import torch
import pandas as pd
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_USED = "clip"

# Set result directory
RESULTS_DIR = "base_rev"

OUTPUT_CSV_PATH = os.path.join(MODEL_USED, RESULTS_DIR, "gender_mean_politician.csv")

BENCHMARK_BRANCHES = ['house', 'mayors', 'senate']
PROMPT_RANGE = range(7, 16)
OCCUPATION_LABELS = [
    "politician", "scientist", "athlete", "teacher", "receptionist",
    "assistant", "salesperson", "actor", "actress"
]

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

# Initialize storage
gender_logits = {"woman": {}, "man": {}}

# Load and collect logits from all branches
for branch in BENCHMARK_BRANCHES:
    input_json_path = os.path.join(RESULTS_DIR, f"base-benchmark-{branch}-detailed_results.json")
    benchmark_txt_path = os.path.join("benchmark-data-v1", f"{branch}.txt")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        model_outputs = json.load(f)

    benchmark_df = pd.read_csv(benchmark_txt_path, delimiter='\t')

    for model_name, image_outputs in model_outputs.items():
        gender_logits["woman"].setdefault(model_name, [])
        gender_logits["man"].setdefault(model_name, [])

        for (_, logits_dict), gender_label in zip(image_outputs.items(), benchmark_df["gender"]):
            selected_logits = [
                logit for idx, (_, logit) in enumerate(logits_dict.items())
                if idx in PROMPT_RANGE
            ]
            gender_logits[gender_label][model_name].append(selected_logits)

# Compute means and write to CSV
output_rows = []

for model_name in gender_logits["woman"].keys():
    for gender in ["woman", "man"]:
        logits_list = gender_logits[gender][model_name]
        if logits_list:  # Avoid division by zero
            mean_logits = torch.tensor(logits_list, dtype=torch.float16).mean(dim=0).to(torch.float32).tolist()
        else:
            mean_logits = [float('nan')] * len(OCCUPATION_LABELS)

        output_rows.append([gender, model_name] + mean_logits[:9])

# Save DataFrame
columns = ["gender", "model"] + OCCUPATION_LABELS
pd.DataFrame(output_rows, columns=columns).to_csv(
    OUTPUT_CSV_PATH,
    index=False,
    encoding="utf-8",
    float_format="%.6f"
)

print(f"Saved CSV file: {OUTPUT_CSV_PATH}")
