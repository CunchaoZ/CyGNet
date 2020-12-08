# CyGNet
This repository reproduces the AAAI'21 paper “Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation Networks” by pytorch.

## Abstract
Knowledge graphs typically grow to contain temporally dynamic facts that model the dynamic relations or interactions of entities along the timeline.
Since such temporal knowledge graphs often suffer from incompleteness, it is important to develop time-aware representation learning models that help infer the missing temporal facts. While the temporal facts are typically evolving, it is noteworthy that, many facts can follow a repeated pattern in history, such as economic crises and diplomatic activities. This indicates that a model could learn much from the known facts in the history. Based on this phenomenon, we propose a new representation learning method for temporal knowledge graphs, namely CyGNet, based on a novel time-aware copy-generation mechanism. CyGNet is not only able to predict future facts from the whole entity vocabulary, but also capable of identifying facts with repetition, and accordingly selecting such future facts based on the known facts in the past. We evaluate the proposed method on the knowledge graph completion task using five benchmark datasets. Extensive experiments demonstrate the effectiveness of CyGNet for predicting future facts with repetition as well as those that are de novo.

## Environment
    python 3.7
    pytorch 1.3.0

## Dataset
There are four datasets (from [RE-NET](https://github.com/INK-USC/RE-Net)): ICEWS18, ICEWS14, GDELT, WIKI, and YAGO. Times of test set should be larger than times of train and valid sets. (Times of valid set also should be larger than times of train set.) Each data folder has 'stat.txt', 'train.txt', 'valid.txt', 'test.txt'.

## Run the experiment
We first get the historical vocabulary.

        python get_historical_vocabulary.py -dataset DATA_NAME
Then, train the model.

        python train.py -dataset ICEWS18 --time-stamp 24 -alpha 0.8 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu 0 --batch-size 1024 --counts 3
        python train.py -dataset ICEWS14 --time-stamp 24 -alpha 0.8 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu 0 --batch-size 1024 --counts 3
        python train.py -dataset GDELT --time-stamp 15 -alpha 0.7 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu 0 --batch-size 1024 --counts 2
        python train.py -dataset WIKI --time-stamp 1 -alpha 0.7 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu 0 --batch-size 1024 --counts 5
        python train.py -dataset YAGO --time-stamp 1 -alpha 0.7 -lr 0.001 --n-epoch 30 --hidden-dim 200 -gpu 0 --batch-size 1024 --counts 5

## Reference
Bibtex:

        @inproceedings{zhu-etal-2021-cygnet,
          title = {Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation Networks},
          author = "Zhu, Cunchao and Chen, Muhao and Fan, Changjun and Cheng, Guangquan and Zhang, Yan",
          booktitle = "AAAI",
          year = "2021",
        }
