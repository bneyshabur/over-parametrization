# The role of over-parametrization in generalization of neural nets

This repository contains the code to train neural nets and compute various measures/norms reported in the following paper:

**[Towards Understanding the Role of Over-Parametrization in Generalization of Neural Networks](https://arxiv.org/abs/1805.12076)**

[Behnam Neyshabur](https://www.neyshabur.net), [Zhiyuan Li](https://sites.google.com/site/invariantorli/), [Srinadh Bhojanapalli](http://ttic.uchicago.edu/~srinadh/), [Yann LeCun](http://yann.lecun.com/), [Nathan Srebro](http://www.ttic.edu/srebro)

## Usage
1. Install *Python 3.6* and *PyTorch 0.4.1*.
2. Clone the repository:
   ```
   git clone https://github.com/bneyshabur/over-parametrization.git
   ```
3. As a simple example, the following command trains a two layer fully connected feedforward network with 1000 hidden units on *CIFAR10* dataset and then computes several measures/norms on the learned network:
   ```
   python main.py --dataset CIFAR10 --nunits 1000
   ```
## Main Inputs Arguments
* `--no-cuda`: disables cuda training
* `--datadir`: path to the directory that contains the datasets (default: datasets)
* `--dataset`: name of the dataset(options: MNIST | CIFAR10 | CIFAR100 | SVHN, default: CIFAR10). If the dataset is not in the desired directory, it will be downloaded.
* `--nunits`: number of hidden units (default: 1024)

## Reported Norms/Measures
After training the network, several norms/measures will be computed and reported on the trained network. Please see the file `measures.py` for explanation of each measure. We also compute and report the following generalization bounds:
* `VC bound`: Generalization bound based on the VC dimension by Harvey et al. 2017
* `L1max bound`: Generalization bound by Bartlett and Mendelson 2002
* `Fro bound`: Generalization bound by Neyshabur et al. 2015
* `Spec_L1 bound`: Generalization bound by Bartlett et al. 2017
* `Spec_Fro bound`: Generalization bound by Neyshabur et al. 2018
* `Our bound`: The Generalization bound proposed in this paper
