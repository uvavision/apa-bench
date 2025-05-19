# UPB Benchmark

Vision-language models like CLIP have become highly capable at matching images and text. To evaluate their associative abilities, societal biases, and knowledge retention, we introduce the APA Benchmark, which consists of portraits of actors, actresses, politicians, and athletes. This paper discusses the implications of the results and examines how model scale affects the evaluated properties. While model performance improves primarily with increased parameter size rather than training data alone, our results reveal that, despite recent advances, vision-language and generative models still risk overlooking essential features or reinforcing gender stereotypes depending on the training data. The proposed benchmark offers a practical dataset for systematically comparing a wide range of vision-language models. Our code, and data are included with this submission and will be released upon publication. 

## Pakage Requirements:
If you need, make conda env.
```
$ conda create -n apa_benchmark python=3.10
```

Install below packages.

- pandas
- pytorch=1.7.1 
- torchvision
- cudatoolkit=11.0 
- ftfy 
- regex 
- tqdm
- transformers
- accelerate

( Gemma 3 is supported starting from transformers 4.50.0 )


## CLIP:

###  Additional Requirements:
- Make sure CLIP is up to date:
  ```
  $ pip install git+https://github.com/openai/CLIP.git
  ```
  
### To compute all scores for img2txt and txt2img run:
```
$ python clip/pks_benchmark.py
$ python clip/pks_benchmark_rev.py 
```

### To cumpute all scores for Easy, Medium and Hard benchmark run:
```
$ python clip/base-benchmark.py
```


## Gemma3 ( Generative VL model ):

### To compute all scores for img2txt and txt2img run:
```
$ python gemma3/pks_benchmark.py
$ python gemma3/pks_benchmark_rev.py
```

### To cumpute all scores for Easy, Medium and Hard benchmark run:
```
$ python gemma3/base-benchmark.py
```


## Evalation:
### To summarize capability:
```
$ python eval/exp1/cal_three_mean_actor.py
$ python eval/exp1/cal_three_mean_politician.py
$ python eval/exp1/cal_three_mean_athletes.py
```

### To summarize bias score:
```
$ python eval/exp2/cal_bias_actor_athlete.py
$ python eval/exp2/cal_bias_politician.py
```

### To summarize bias score:
```
$ python eval/exp3/cal_image_score_mean.py
$ python eval/exp3/cal_text_score_mean.py
```
