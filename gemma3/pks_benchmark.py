import os
import csv
import json
import pandas as pd
from PIL import Image
import numpy as np
from transformers import TorchAoConfig, AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer
import torch.nn.functional as F 
import torch

# Calculate the Image Score

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_USED = "gemma3"

# Set result directory
RESULTS_DIR = "pks_res"

# Branches to evaluate
BRANCHES = ['house', 'mayors', 'senate', 'actors', 'athletes']

models_list = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it"
]

for branch in BRANCHES:

    # Target data to be used for the testing
    target_data = os.path.join('benchmark-data-v1/' + branch + '.txt')
    image_folder = os.path.join('benchmark-data-v1/' + branch)

    # Read image metadata
    pdframe = pd.read_csv(target_data, delimiter = '\t')

    # Generate prompts using names
    names_captions = [f"this is a photo of {name}" for name in pdframe['names']]
    
    detailed_results = {}
    
    for model_name in models_list:
        print("------------------------------------------------------------\n")
        print(f"Processing with model: {model_name} using {branch} data\n")
        print("------------------------------------------------------------\n")

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()

        processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        out_path = os.path.join(MODEL_USED, RESULTS_DIR, f'pks_benchmark_{branch}_detailed_results.json')

        # Load if the file already exists
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                detailed_results = json.load(f)
        else:
            detailed_results = {}

        people_results = {}
        for index, row in pdframe.iterrows():
            person_name = row['names']
            image_path = os.path.join(image_folder, row['filename'])

            if index % 10 == 0:
                print(f"{model_name} - Progress: {index} processed.")
            
            score_res = []
            for caption in names_captions:
                prompt = f"Indicate if the following statement is true or false, and please only output 'true' or 'false': {caption}"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(model.device, dtype=torch.bfloat16)
                
                with torch.inference_mode():
                    outputs = model(**inputs)
                    logits = outputs.logits

                true_token_id = tokenizer.convert_tokens_to_ids("true")
                true_logits = logits[..., true_token_id]
                last_position_logit = true_logits[0, -1]
                score_res.append(last_position_logit.item())
            
            # Apply softmax to score_res here to convert into probabilistic scores
            score_res = F.softmax(torch.tensor(score_res), dim=0).tolist()

            res_per_person = {}
            for ii in range(len(names_captions)):
                res_per_person[names_captions[ii]] = score_res[ii]
            people_results[person_name] = res_per_person

        detailed_results[model_name] = people_results

        with open(out_path, "w") as out_file:
            json.dump(detailed_results, out_file, indent=4)
