import os
import csv
import json
import clip
import torch
import pandas as pd
from PIL import Image
import open_clip

# Get path to local directory.
base_path = os.path.dirname(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"
out_folder = "open_clip_res"

run_model_type = "clip"
clip_models = clip.available_models()
open_clip_available_models = list(open_clip.pretrained._PRETRAINED.keys())
open_clip_models = { 'laion400m_e32': ['ViT-B-32-quickgelu', 'ViT-B-16', 'ViT-L-14'], 
                    'laion2b_s32b_b82k': ['ViT-L-14'],
                    'laion2b_s32b_b79k': ['ViT-H-14'],
                    'laion2b_s34b_b79k': ['ViT-B-32']
                    }

# Branch of government to test.
for branch in ['house', 'mayors', 'senate']:

    # Target data to be used for the testing.
    target_data = os.path.join(base_path, 'benchmark-data-v1/' + branch + '.txt')
    image_folder = os.path.join(base_path, 'benchmark-data-v1/' + branch)

    # Read image metadata.
    pdframe = pd.read_csv(target_data, delimiter = '\t')

    # Generate prompts using names
    names_captions = [f"this is a photo of {name}" for name in pdframe['names']]
    
    detailed_results = {}
    if run_model_type == "open_clip":
        for model_data, model_arch in open_clip_models.items():
            for model_name in model_arch:
                print (model_name, model_data)
                target_model = f"{model_name} with {model_data}"
                
                model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=model_data)
                tokenizer = open_clip.get_tokenizer(model_name)
                model = model.to(device)
                
                tokenized_prompts = tokenizer(names_captions).to(device)
                
                # Read and preprocess images.
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
                        score_res = (100.0 * image_features @ text_features.T).softmax(dim=-1).detach().cpu()

                    res_per_person = {}
                    for ii in range (len(names_captions)):
                        res_per_person[names_captions[ii]] = score_res[0, ii].item()
                    people_results[person_name] = res_per_person
                    
                detailed_results[target_model] = people_results

    elif run_model_type == "clip":
        tokenized_prompts = clip.tokenize(names_captions).to(device)
        for target_model in clip_models:
            print('Probing %s...' % target_model)

            model, preprocess = clip.load(target_model, device)
            
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
                for ii in range (len(names_captions)):
                    res_per_person[names_captions[ii]] = score_res[0, ii].item()
                people_results[person_name] = res_per_person

            detailed_results[target_model] = people_results


    # the json file where the output must be stored
    out_file = open(f'{run_model_type}_res/pks_benchmark_{branch}_detailed_results.json', "w")
    json.dump(detailed_results, out_file, indent = 4)
    out_file.close()