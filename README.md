# The APA Benchmark: A People-centric Benchmark for Testing Vision-Language Models

We introduce the APA Benchmark, consisting of images of Actors, Politicians, and Athletes paired with a series of text prompts. Our benchmark serves as a tool for practitioners and researchers who are considering VLMs for people-centric tasks. Our benchmark tests VLM for their association ability with respect to pictures of public figures in three domains: Athletics, Politics and Acting. We issue text prompts against photos of famous people in each of these domains and provide a score for VLMs according to their matching capabilities. Images are mostly sourced from Wikimedia Commons and Wikipedia which means they are either in the public domain or have a friendly license to redistribute. Other metadata is manually curated from Wikipedia or from official sources. 

---

## Environment Setup

### 1. Create Conda Environment (Optional)

If needed, create a new conda environment with Python 3.10:

```bash
conda create -n apa_benchmark python=3.10
conda activate apa_benchmark
````

### 2. Install Required Packages

Install the following Python packages:

* pandas
* pytorch==1.7.1
* torchvision
* cudatoolkit==11.0
* ftfy
* regex
* tqdm
* transformers
* accelerate

>  **Note:** Gemma 3 is supported starting from `transformers==4.50.0`.

---

## CLIP Benchmark

### Additional Installation

Make sure you have the latest version of CLIP installed:

```bash
pip install git+https://github.com/openai/CLIP.git
```

### Running Benchmarks

#### Compute scores by prompt type: category, occupation, and specialty

```bash
python clip/base-benchmark.py
```

#### Compute scores for Image-to-Text tasks

```bash
python clip/pks_benchmark.py
```

#### Compute scores for Text-to-Image tasks

```bash
python clip/pks_benchmark_rev.py
```

---

## Gemma 3 (Generative Vision-Language Model)

### Running Benchmarks

#### Compute scores by prompt type: category, occupation, and specialty

```bash
python gemma3/base-benchmark.py
```

#### Compute scores for Image-to-Text tasks

```bash
python gemma3/pks_benchmark.py
```

#### Compute scores for Text-to-Image tasks

```bash
python gemma3/pks_benchmark_rev.py
```

---

## Evaluation Scripts

These scripts summarize benchmark results from different analytical perspectives.

### 1. Basic associative abilities

```bash
python eval/exp1/cal_three_mean_actor.py
python eval/exp1/cal_three_mean_politician.py
python eval/exp1/cal_three_mean_athletes.py
```

### 2. Influence of societal biases

```bash
python eval/exp2/cal_bias_actor_athlete.py
python eval/exp2/cal_bias_politician.py
```

### 3. Identity recognition capability

```bash
python eval/exp3/cal_image_score_mean.py
python eval/exp3/cal_text_score_mean.py
```

---

## 📚 Citation

If you use this work, please cite us using the following BibTeX entry:

```bibtex
@misc{apabench,
  author       = {Yuri Ishitoya and Veronica Flores and Ziyan Yang and Paola Cascante-Bonilla and Vicente Ordonez},
  title        = {The APA Benchmark: A People-centric Benchmark for Testing Vision-Language Models},
  year         = {2025},
  howpublished = {\url{https://github.com/uvavision/vislang-apa-benchmark}}
}
```

---

## License

This project is licensed under the MIT License.
