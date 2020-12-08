CyGNet
===
This repository reproduces the AAAI'21 paper “Learning from History: Modeling Temporal Knowledge Graphs with Sequential Copy-Generation Networks” by pytorch.

## Abstract
Knowledge graphs typically grow to contain temporally dynamic facts that model the dynamic relations or interactions of entities along the timeline.
Since such temporal knowledge graphs often suffer from incompleteness, it is important to develop time-aware representation learning models that help infer the missing temporal facts. While the temporal facts are typically evolving, it is noteworthy that, many facts can follow a repeated pattern in history, such as economic crises and diplomatic activities. This indicates that a model could learn much from the known facts in the history. Based on this phenomenon, we propose a new representation learning method for temporal knowledge graphs, namely CyGNet, based on a novel time-aware copy-generation mechanism. CyGNet is not only able to predict future facts from the whole entity vocabulary, but also capable of identifying facts with repetition, and accordingly selecting such future facts based on the known facts in the past. We evaluate the proposed method on the knowledge graph completion task using five benchmark datasets. Extensive experiments demonstrate the effectiveness of CyGNet for predicting future facts with repetition as well as those that are de novo.

## Environment
python 3.7
pytorch 1.3.0

## Dataset

## Run the experiment

