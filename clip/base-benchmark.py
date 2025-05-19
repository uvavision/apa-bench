import os
import csv
import json
import clip
import torch
import pandas as pd
import numpy as np
from PIL import Image

BASE_PATH = os.path.dirname(__file__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_USED = "clip"

# Set result directory
RESULTS_DIR = "base_res"

# Branches to evaluate
BRANCHES = ['house', 'mayors', 'senate', 'actors', 'athletes']

MODELS = clip.available_models()

for branch in BRANCHES:
    # Load image metadata and prompts
    target_file = os.path.join(f'benchmark-data-v1/{branch}.txt')
    image_folder = os.path.join(f'benchmark-data-v1/{branch}')
    prompt_file = os.path.join(f'base-prompts-{branch}.csv')

    # Output paths
    output_txt_path = os.path.join(MODEL_USED, RESULTS_DIR, f'base-benchmark-{branch}-results.txt')
    output_json_path = os.path.join(MODEL_USED, RESULTS_DIR, f'base-benchmark-{branch}-detailed_results.json')

    df_images = pd.read_csv(target_file, delimiter='\t')
    df_prompts = pd.read_csv(prompt_file, delimiter='|')

    tokenized_prompts = clip.tokenize(df_prompts['Prompt'].tolist()).to(DEVICE)
    prompt_difficulties = df_prompts['Difficulty'].tolist()

    detailed_results = {}

    for model_name in MODELS:
        print("------------------------------------------------------------\n")
        print(f"Processing with model: {model_name} using {branch} data\n")
        print("------------------------------------------------------------\n")
        
        model, preprocess = clip.load(model_name, DEVICE)

        logits_list = []
        results_per_image = {}

        for idx, row in df_images.iterrows():
            image_path = os.path.join(image_folder, row['filename'])
            image = Image.open(image_path)
            image_input = preprocess(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits_image, _ = model(image_input, tokenized_prompts)

                # Split logits by difficulty level
                easy_mask = [d == 'Easy' for d in prompt_difficulties]
                medium_mask = [d == 'Medium' for d in prompt_difficulties]
                hard_mask = [d == 'Hard' for d in prompt_difficulties]

                logits_easy = logits_image[0, easy_mask].softmax(dim=-1).cpu()
                logits_medium = logits_image[0, medium_mask].softmax(dim=-1).cpu()
                logits_hard = logits_image[0, hard_mask].softmax(dim=-1).cpu()

                # Concatenate scores across difficulty levels
                combined_logits = torch.cat((logits_easy, logits_medium, logits_hard), dim=0)
                logits_list.append(combined_logits.unsqueeze(0))

                # Save results for each prompt
                results_per_image[idx] = {
                    df_prompts['Prompt'][i]: combined_logits[i].item()
                    for i in range(len(df_prompts['Prompt']))
                }

        # Store detailed results per model
        detailed_results[model_name] = results_per_image

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=4)

        # Compute mean and std of logits
        all_logits = torch.cat(logits_list, dim=0)
        logits_mean = all_logits.mean(dim=0).to(torch.float32)
        logits_std = all_logits.std(dim=0).to(torch.float32)

        # Save to DataFrame
        df_prompts[f'{model_name}-mean'] = logits_mean.cpu().numpy()
        # (optional: store std too if needed)
        # df_prompts[f'{model_name}-std'] = logits_std.cpu().numpy()

        print(df_prompts)

    # Save results
    df_prompts.to_csv(output_txt_path, sep='\t', quoting=csv.QUOTE_NONNUMERIC)