import os
import csv
import json
import clip
import torch
import pandas as pd
from PIL import Image

# Calculate the Image Score

# Get path to local directory
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_USED = "clip"

# Set result directory
RESULTS_DIR = "pks_res"

BRANCHES = ['house', 'mayors', 'senate', 'actors', 'athletes']

clip_models = clip.available_models()

for branch in BRANCHES:

    # Target data to be used for the testing
    target_data = os.path.join('benchmark-data-v1/' + branch + '.txt')
    image_folder = os.path.join('benchmark-data-v1/' + branch)
    output_path = os.path.join(MODEL_USED, RESULTS_DIR, f'pks_benchmark_{branch}_detailed_results.json')

    # Read image metadata
    pdframe = pd.read_csv(target_data, delimiter='\t')

    # Generate prompts using names
    names_captions = [f"this is a photo of {name}" for name in pdframe['names']]
    
    detailed_results = {}
    tokenized_prompts = clip.tokenize(names_captions).to(device)

    for model_name in clip_models:

        print("------------------------------------------------------------\n")
        print(f"Processing with model: {model_name} using {branch} data\n")
        print("------------------------------------------------------------\n")

        model, preprocess = clip.load(model_name, device)
        
        people_results = {}
        for index, row in pdframe.iterrows():
            image_path = os.path.join(image_folder, row['filename'])
            img = Image.open(image_path)
            img_preprocessed = preprocess(img).unsqueeze(0).to(device)
            person_index = row['Unnamed: 0']
            person_name = row['names']

            with torch.no_grad():
                logits_img, logits_text = model(img_preprocessed, tokenized_prompts)
                score_res = logits_img.softmax(dim=-1).cpu().numpy()

            res_per_person = {}
            for ii in range(len(names_captions)):
                res_per_person[names_captions[ii]] = score_res[0, ii].item()
            people_results[person_name] = res_per_person

        detailed_results[model_name] = people_results

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=4)