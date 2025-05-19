import os
import csv
import json
import clip
import torch
import pandas as pd
from PIL import Image

# Calculate the Text Score

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_USED = "clip"

# Set result directory
RESULTS_DIR = "pks_rev_res"

BRANCHES = ['house', 'mayors', 'senate', 'actors', 'athletes']

clip_models = clip.available_models()

for branch in BRANCHES:

    # Target data to be used for the testing
    target_data = os.path.join('benchmark-data-v1/' + branch + '.txt')
    image_folder = os.path.join('benchmark-data-v1/' + branch)
    output_path = os.path.join(MODEL_USED, RESULTS_DIR, f'pks_benchmark_rev_{model_name}_{branch}_detailed_results.json')

    # Read image metadata
    pdframe = pd.read_csv(target_data, delimiter = '\t')

    # Generate prompts using names
    names_captions = [f"this is a photo of {name}" for name in pdframe['names']]
    
    for model_name in clip_models: 
        print("------------------------------------------------------------\n")
        print(f"Processing with model: {model_name} using {branch} data\n")
        print("------------------------------------------------------------\n")

        model, preprocess = clip.load(model_name, device)

        # Rename for saving to file
        model_name = model_name.replace("/", "-")

        # Get all images and preprocess them
        img_preprocessed = torch.cat([
            preprocess(Image.open(os.path.join(image_folder, ff))).unsqueeze(0).to(device) 
            for ff in pdframe['filename']
        ], dim=0)
        
        people_results = {}
        for person_idx, person_name_caption in enumerate(names_captions):
            
            tokenized_prompts = clip.tokenize(person_name_caption).to(device)
            person_name = pdframe['names'][person_idx]

            with torch.no_grad():
                image_features = model.encode_image(img_preprocessed)
                text_features = model.encode_text(tokenized_prompts)
                logits_img, logits_text = model(img_preprocessed, tokenized_prompts)
                score_res = logits_text.softmax(dim=-1).cpu().numpy()

            res_per_person = {}
            for ii in range(len(names_captions)):
                res_per_person[names_captions[ii]] = score_res[0, ii].item()
            people_results[person_name] = res_per_person

        # Save results to JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(people_results, f, indent=4)
