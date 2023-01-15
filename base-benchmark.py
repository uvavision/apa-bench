import os
import csv
import clip
import torch
import pandas as pd
from PIL import Image

# Get path to local directory.
base_path = os.path.dirname(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Branch of government to test.
branch = 'mayors'
models = clip.available_models()

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
tokenized_prompts = clip.tokenize(prompts['Prompt'].to_list()).to(device)
prompt_difficulties = prompts['Difficulty'].to_list()

# Target model to be tested.
for target_model in models:
    print('Probing %s...' % target_model)

    # Load the model
    model, preprocess = clip.load(target_model, device)

    # Read and preprocess images.
    logits_list = list()
    for index, row in pdframe.iterrows():
        image_path = os.path.join(image_folder, row['filename'])
        img = Image.open(image_path)
        img_preprocessed = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_img, logits_text = model(img_preprocessed, tokenized_prompts)
            logits_easy = logits_img[0, [difficulty == 'Easy' for difficulty in prompt_difficulties]]
            logits_medium = logits_img[0, [difficulty == 'Medium' for difficulty in prompt_difficulties]]
            logits_hard = logits_img[0, [difficulty == 'Hard' for difficulty in prompt_difficulties]]
            
            logits_easy = logits_easy.softmax(dim=-1).detach().cpu()
            logits_medium = logits_medium.softmax(dim=-1).detach().cpu()
            logits_hard = logits_hard.softmax(dim=-1).detach().cpu()

            logits_img = torch.cat((logits_easy, logits_medium, logits_hard), 0)
            logits_list.append(logits_img.unsqueeze(0))

    logits_mean = torch.cat(logits_list, 0).mean(dim = 0)
    logits_std = torch.cat(logits_list, 0).std(dim = 0)

    prompts[target_model + '-mean'] = logits_mean
    prompts[target_model + '-std'] = logits_std
    print(prompts)

prompts.to_csv('base-benchmark-'+ branch + '-results.txt', sep = '\t', quoting = csv.QUOTE_NONNUMERIC)
        
