import json
import torch
import pandas as pd
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_USED = "clip"

# Set result directory
RESULTS_DIR = "base_rev"

# Prompt index range for occupation-related prompts
PROMPT_RANGE = range(7, 16)

# Output column names
OCCUPATION_LABELS = [
    "politician", "scientist", "athlete", "teacher", "receptionist",
    "assistant", "salesperson", "actor", "actress"
]

# Main processing loop
for branch in ['actors', 'athletes']:
    input_json_path = os.path.join(MODEL_USED, RESULTS_DIR, f"base-benchmark-{branch}-detailed_results.json")
    benchmark_txt_path = os.path.join("benchmark-data-v1", f"{branch}.txt")
    output_csv_path = os.path.join(MODEL_USED, RESULTS_DIR, f"gender_mean_{branch}.csv")

    # Load model output
    with open(input_json_path, 'r', encoding='utf-8') as f:
        model_outputs = json.load(f)

    # Load gender labels
    benchmark_df = pd.read_csv(benchmark_txt_path, delimiter='\t')

    output_rows = []

    for model_name, image_outputs in model_outputs.items():
        woman_logits, man_logits = [], []

        for (_, logits_dict), gender_label in zip(image_outputs.items(), benchmark_df["gender"]):
            selected_logits = [
                logit for idx, (_, logit) in enumerate(logits_dict.items())
                if idx in PROMPT_RANGE
            ]

            if gender_label == "woman":
                woman_logits.append(selected_logits)
            else:
                man_logits.append(selected_logits)

        # Compute mean logits for each gender group
        woman_mean = torch.tensor(woman_logits, dtype=torch.float16).mean(dim=0).to(torch.float32).tolist()
        man_mean = torch.tensor(man_logits, dtype=torch.float16).mean(dim=0).to(torch.float32).tolist()

        # Append to final CSV rows
        output_rows.append(["woman", model_name] + woman_mean[:9])
        output_rows.append(["man", model_name] + man_mean[:9])

    # Save to CSV
    columns = ["gender", "model"] + OCCUPATION_LABELS
    pd.DataFrame(output_rows, columns=columns).to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8",
        float_format="%.6f"
    )

    print(f"Saved CSV file: {output_csv_path}")
