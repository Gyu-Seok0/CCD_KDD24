# Continual Collaborative Distillation for Recommender System (CCD)

### 1. Overview
This repository provides the source code of our paper: Continual Collaborative Distillation for Recommender System (CCD).

This paper introduces an integrated framework combining knowledge distillation and continual learning for practical recommender systems. This approach enables collaborative evolution between teacher and student models in dynamic environments.

<img src="./figure/method.png">

### 2. Usage

Each data block should be trained in the following order.

```
1. Student Update
python -u S_update.py --d Yelp -m LightGCN_1 --tt 1 --BD --US --UP --ab 50 --ss 1 --ps 0 --sw 1.0 --pw 0.1 --s --max_epoch 10

2. Teacher Update
python -u T_update.py --d Yelp --student LightGCN_1 --teacher LightGCN_1 --tt 1 --BD --UCL --US --UP --ss 1 --ps 3 --cs 5 --s --max_epoch 10

3. Ensemble
python -u Ensemble.py --d Yelp --tt 1 --s

4. KD
python KD.py --d Yelp -m LightGCN_1 --tt 1 --s --max_epoch 10
```

You can find an overview of our framework's entire execution in run.sh.

### 3. Environment
The codes are written in Python 3.9.0.
* pytorch==1.13.1
* pytorch-cuda=11.6
* numpy=1.24.3

<br> Details see env.yml.
