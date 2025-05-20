# The APA Benchmark: A People-centric Benchmark for Testing Vision-Language Models

We introduce the APA Benchmark, consisting of images of Actors, Politicians, and Athletes paired with a series of text prompts. Our benchmark serves as a tool for practitioners and researchers who are considering VLMs for people-centric tasks. We demonstrate the usefulness of our benchmark by systematically probing a large variety of modern VLMs for their associative abilities in this domain. 
We discuss the implications of these experiments and examine how model scale affects their basic associative abilities, influence from societal biases and capacity for identity recognition. 

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

#### Compute Scores for Image-to-Text and Text-to-Image Tasks:

```bash
python clip/pks_benchmark.py
python clip/pks_benchmark_rev.py
```

#### Compute Scores by Difficulty Level (Easy, Medium, Hard):

```bash
python clip/base-benchmark.py
```

---

## Gemma 3 (Generative Vision-Language Model)

### Running Benchmarks

#### Compute Scores for Image-to-Text and Text-to-Image Tasks:

```bash
python gemma3/pks_benchmark.py
python gemma3/pks_benchmark_rev.py
```

#### Compute Scores by Difficulty Level (Easy, Medium, Hard):

```bash
python gemma3/base-benchmark.py
```

---

## Evaluation Scripts

These scripts summarize the benchmark results from different perspectives: Bias Assessment and Identity Recognition

### 1. Capability Summary (Category-Wise Mean Scores)

```bash
python eval/exp1/cal_three_mean_actor.py
python eval/exp1/cal_three_mean_politician.py
python eval/exp1/cal_three_mean_athletes.py
```

### 2. Bias Summary (Score Differences Between Groups)

```bash
python eval/exp2/cal_bias_actor_athlete.py
python eval/exp2/cal_bias_politician.py
```

### 3. Modality-Wise Summary (Image / Text Score Means)

```bash
python eval/exp3/cal_image_score_mean.py
python eval/exp3/cal_text_score_mean.py
```

---

## License

This project is licensed under the MIT License.
