# UPB Benchmark

Vision-language models such as CLIP that are able to compute matching scores between images and text have become extremely capable. We propose the US Politicians Benchmark (UPB) to probe the capabilities in the associations computed by these models, the societal biases induced by these associations, and the capacity of these models to retain knowledge. The UPB Benchmark consists of public domain portraits of major politicians in the United States from the Senate, the House of Representatives and mayors of the most populated cities. We discuss some of the implications of our results and discover the role of scale in each of the properties targeted in our study. Similar as in the pure textual domain, there are capabilities in vision-language models that seem to emerge only in the largest models. As more variants of vision-language models are trained on publicly available data, we expect that our benchmark will be an easy test to replicate. Our code, and data are included with this submission and will be released upon publication.

## Requirements:
- Make sure CLIP and Open CLIP are up to date:
  ```
  $ pip install git+https://github.com/openai/CLIP.git
  $ pip install --upgrade open_clip_torch
  $ git clone [this repo]
  $ mkdir clip_res
  $ mkdir open_clip_res
  ```
Additional packages:
- pandas
- pytorch=1.7.1 
- torchvision
- cudatoolkit=11.0 
- ftfy 
- regex 
- tqdm
___

### To compute all scores for img2txt and txt2img run:
```
$ python pks_benchmark.py
$ python pks_benchmark_rev.py
```

### To cumpute all scores for Easy, Medium and Hard benchmark run:
```
$ python base-benchmark.py
$ python open_clip_bench.py
```
