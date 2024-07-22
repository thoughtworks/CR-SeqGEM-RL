[![CR-SeqGEM-RL](https://github.com/thoughtworks/CR-SeqGEM-RL/actions/workflows/python-app_conda.yml/badge.svg)](https://github.com/thoughtworks/CR-SeqGEM-RL/actions/workflows/python-app_conda.yml)

# CR-SeqGEM-RL: Cellular Reprogramming by optimizing Sequential Gene Expression Modulation using Reinforcement Learning
A computational method to predict sequential gene expression modulation for targeted cellular reprogramming is presented. The method integrates: (1) a Boolean model of the concerned gene regulatory network (GRN); and (2) a reinforcement learning (RL) based model for optimization. The Boolean model is used to capture the dynamic behavior of the GRN and to understand how the gene expression modulation alters its behavior. RL model is used to optimize sequential decision-making of predicting the suitable sequence of gene expression modulation.

For more details, refer to the preprint of this study: [Link](https://www.biorxiv.org/content/10.1101/2024.03.19.585672v1)


## Prerequisites
- Operating System: Linux (Currently tested on Ubuntu 20.04 and 22.04)
- conda

pyboolnet dependencies:
- clasp
- gringo


## Installation
```
git clone <THIS_REPO>

# The following command force creates a conda environment with the name: cr-seqgem-rl. Any existing environment with the same name will be overwritten.
conda env create -f environment.yml --force

conda activate cr-seqgem-rl
```


## Usage
The code can be executed with an example Boolean network model corresponding to the core gene regulatory network of early heart development in mouse [[Ref-1](https://pubmed.ncbi.nlm.nih.gov/23056457/), [Ref-2](https://ieeexplore.ieee.org/abstract/document/8704946)]. For more details, please refer to the preprint of this study ([Link](https://www.biorxiv.org/content/10.1101/2024.03.19.585672v1)).

To train the model, run:
```
train.py
```
\
To use the trained model for inferencing, run:
```
infer.py
```
