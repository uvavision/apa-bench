import os
import csv
import json
# import clip
import torch
import pandas as pd
from PIL import Image
import open_clip

# Get path to local directory.
base_path = os.path.dirname(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"
out_folder = "open_clip_res"

# Branch of government to test.
for branch in ['house', 'mayors', 'senate']:
    # models = clip.available_models()

    # Target data to be used for the testing.
    target_data = os.path.join(base_path, 'benchmark-data-v1/' + branch + '.txt')
    image_folder = os.path.join(base_path, 'benchmark-data-v1/' + branch)

    # Load prompts
    prompt_data = os.path.join(base_path, 'base-prompts.csv')

    # Read image metadata.
    pdframe = pd.read_csv(target_data, delimiter = '\t')

    # Read prompt texts.
    prompts = pd.read_csv(prompt_data, delimiter = '|')

    # Tokenize the prompt texts.
    # tokenized_prompts = clip.tokenize(prompts['Prompt'].to_list()).to(device)
    prompt_difficulties = prompts['Difficulty'].to_list()

    difficulties = {}
    for idx, difficulty in enumerate(prompt_difficulties):
        if difficulty in difficulties:
            difficulties[difficulty].append(idx)
        else:
            difficulties[difficulty] = [idx]

    open_clip_available_models = list(open_clip.pretrained._PRETRAINED.keys())
    open_clip_models = { 'laion400m_e32': ['ViT-B-32-quickgelu', 'ViT-B-16', 'ViT-L-14'], 
                        'laion2b_s32b_b82k': ['ViT-L-14'],
                        'laion2b_s32b_b79k': ['ViT-H-14'],
                        'laion2b_s34b_b79k': ['ViT-B-32']
                        }

    detailed_results = {}
    for model_data, model_arch in open_clip_models.items():
        for model_name in model_arch:
            print (model_name, model_data)
            target_model = f"{model_name} with {model_data}"
            
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_data)
            tokenizer = open_clip.get_tokenizer(model_name)
            model = model.to(device)
            
            tokenized_prompts = tokenizer(prompts['Prompt'].to_list()).to(device)
            
            # Read and preprocess images.
            logits_list = list()
            people_results = {}
            for index, row in pdframe.iterrows():
                image_path = os.path.join(image_folder, row['filename'])
                img_preprocessed = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                person_index = row['Unnamed: 0']
                person_name = row['names']

                with torch.no_grad():
                    # logits_img, logits_text = model(img_preprocessed, tokenized_prompts)
                    image_features = model.encode_image(img_preprocessed)
                    text_features = model.encode_text(tokenized_prompts)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    score_res = (100.0 * image_features @ text_features.T)

                    logits_easy = score_res[0, difficulties['Easy']]
                    logits_medium = score_res[0, difficulties['Medium']]
                    logits_hard = score_res[0, difficulties['Hard']]

                    logits_easy = logits_easy.softmax(dim=-1).detach().cpu()
                    logits_medium = logits_medium.softmax(dim=-1).detach().cpu()
                    logits_hard = logits_hard.softmax(dim=-1).detach().cpu()

                    logits_img = torch.cat((logits_easy, logits_medium, logits_hard), 0)
                    logits_list.append(logits_img.unsqueeze(0))

                res_per_person = {}
                for ii in range (len(prompts['Prompt'])):
                    res_per_person[prompts['Prompt'][ii]] = logits_img[ii].item()
                people_results[person_name] = res_per_person
                
            detailed_results[target_model] = people_results

            logits_mean = torch.cat(logits_list, 0).mean(dim = 0)
            logits_std = torch.cat(logits_list, 0).std(dim = 0)

            prompts[target_model + '-mean'] = logits_mean
            prompts[target_model + '-std'] = logits_std
            print(prompts)
            
    prompts.to_csv(f'{out_folder}/base-benchmark-{branch}-results.txt', sep = '\t', quoting = csv.QUOTE_NONNUMERIC)

    # the json file where the output must be stored
    out_file = open(f'{out_folder}/base-benchmark-{branch}-detailed_results.json', "w")
    json.dump(detailed_results, out_file, indent = 4)
    out_file.close()