import csv
import json
import pandas as pd
from PIL import Image
import numpy as np
from transformers import TorchAoConfig, AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_USED = "gemma3"

# Set result directory
RESULTS_DIR = "base_res"

# Branches to evaluate
BRANCHES = ['house', 'mayors', 'senate', 'actors', 'athletes']

for branch in BRANCHES:
    
    # Target data to be used for the testing
    target_data = os.path.join('benchmark-data-v1/' + branch + '.txt')
    image_folder = os.path.join('benchmark-data-v1/' + branch)
    prompt_data = os.path.join(f'base-prompts-{branch}.csv')

    # Output paths
    output_txt_path = os.path.join(MODEL_USED, RESULTS_DIR, f'base-benchmark-{branch}-results.txt')
    output_json_path = os.path.join(MODEL_USED, RESULTS_DIR, f'base-benchmark-{branch}-detailed_results.json')
    
    # Read image metadata
    pdframe = pd.read_csv(target_data, delimiter='\t')
    prompts = pd.read_csv(prompt_data, delimiter='|')
    
    # Difficulty categories
    prompt_difficulties = prompts['Difficulty'].to_list()
    difficulties = {}
    for idx, difficulty in enumerate(prompt_difficulties):
        difficulties.setdefault(difficulty, []).append(idx)
    
    models_list = [
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
    ]

    if os.path.exists(output_json_path):
        with open(output_json_path, "r") as f:
            detailed_results = json.load(f)
    else:
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

        logits_list = []
        res_per_person = {}

        # For each image
        for index, row in pdframe.iterrows():
            score_res = []
            img_path = image_folder + '/' + row['filename']
            if index % 10 == 0:
                print(f"{model_name},Progress: {img_path} processed.")

            for prompt_text in prompts['Prompt'].to_list():
                prompt = f"Indicate if the following statement is true or false, and please only output 'true' or 'false': {prompt_text}"

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
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

            score_res_tensor = torch.tensor(score_res)
            logits_easy = score_res_tensor[difficulties['Easy']].softmax(dim=-1).cpu().numpy().tolist()
            logits_medium = score_res_tensor[difficulties['Medium']].softmax(dim=-1).cpu().numpy().tolist()
            logits_hard = score_res_tensor[difficulties['Hard']].softmax(dim=-1).cpu().numpy().tolist()

            logits_img = logits_easy + logits_medium + logits_hard
            logits_list.append(torch.tensor(logits_img).unsqueeze(0))

            res_per_person[index] = {}
            for ii in range(len(prompts['Prompt'])):
                res_per_person[index][prompts['Prompt'][ii]] = logits_img[ii]
        
            detailed_results[model_name] = res_per_person

            with open(output_json_path, "w") as out_file:
                json.dump(detailed_results, out_file, indent=4)
        
        # Convert the nested lists to tensors
        logits_tensor_list = [logits.clone().detach().to(torch.float32) for logits in logits_list]

        # Concatenate along the 0th dimension
        logits_mean = torch.stack(logits_tensor_list).mean(dim=0)

        # Convert to float32 and add to DataFrame as numpy
        logits_mean = logits_mean.squeeze(0)
        np.set_printoptions(suppress=True, precision=6)

        prompts[model_name + '-mean'] = logits_mean.cpu().numpy()
        
    df_prompts.to_csv(output_txt_path, sep='\t', quoting=csv.QUOTE_NONNUMERIC)
