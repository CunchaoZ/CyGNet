# CyGNet
This repository reproduces the AAAI'21 paper “Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation Networks” by pytorch.

## Abstract
Large knowledge graphs often grow to store temporal facts that model the dynamic relations or interactions of entities along the timeline. Since such temporal knowledge graphs often suffer from incompleteness, it is important to develop time-aware representation learning models that help to infer the missing temporal facts. While the temporal facts are typically evolving, it is observed that many facts often show a repeated pattern along the timeline, such as economic crises and diplomatic activities. This observation indicates that a model could potentially learn much from the known facts appeared in history. To this end, we propose a new representation learning model for temporal knowledge graphs, namely CyGNet, based on a novel time-aware copy-generation mechanism. CyGNet is not only able to predict future facts from the whole entity vocabulary, but also capable of identifying facts with repetition and accordingly predicting such future facts with reference to the known facts in the past. We evaluate the proposed method on the knowledge graph completion task using five benchmark datasets. Extensive experiments demonstrate the effectiveness of CyGNet for predicting future facts with repetition as well as de novo fact prediction.

## Environment
    python 3.7
    pytorch 1.3.0

## Dataset
There are five datasets (from [RE-NET](https://github.com/INK-USC/RE-Net)): ICEWS18, ICEWS14, GDELT, WIKI, and YAGO. Times of test set should be larger than times of train and valid sets. (Times of valid set also should be larger than times of train set.) Each data folder has 'stat.txt', 'train.txt', 'valid.txt', 'test.txt'.

## Run the experiment
We first get the historical vocabulary.

    python get_historical_vocabulary.py --dataset DATA_NAME
Then, train the model and test.

    python train_test.py --dataset ICEWS18 --time-stamp 24 -alpha 0.8 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu 0 --batch-size 1024 --counts 3
    python train_test.py --dataset ICEWS14 --time-stamp 24 -alpha 0.8 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu 0 --batch-size 1024 --counts 3
    python train_test.py --dataset GDELT --time-stamp 15 -alpha 0.7 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu 0 --batch-size 1024 --counts 2
    python train_test.py --dataset WIKI --time-stamp 1 -alpha 0.7 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu 0 --batch-size 1024 --counts 5
    python train_test.py --dataset YAGO --time-stamp 1 -alpha 0.7 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu 0 --batch-size 1024 --counts 5

## Reference
Bibtex:

    @inproceedings{zhu-etal-2021-cygnet,
      title = {Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation Networks},
      author = "Zhu, Cunchao and Chen, Muhao and Fan, Changjun and Cheng, Guangquan and Zhang, Yan",
      booktitle = "AAAI",
      year = "2021"
    }
