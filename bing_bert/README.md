# BERT pre-training

This repo contains DeepSpeed's version of BERT for pre-training.

Using DeepSpeed's optimized transformer kernels as the building block,
we were able to achieve the fastest BERT training record: 44 minutes on 1,024
NVIDIA V100 GPUs, compared with the previous best published result of 67 minutes on
the same number and generation of GPUs.

* Brief overview and more details, see our [press release](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/).
* Detailed technology deep dive, see our [blog post](https://www.deepspeed.ai/news/2020/05/27/fastest-bert-training.html).
* Tutorial on how to reproduce our results, see our [BERT pre-training tutorial](https://www.deepspeed.ai/tutorials/bert-pretraining/).
* The source code for our transformer kernels can be found in the [DeepSpeed repo](https://github.com/microsoft/deepspeed).
