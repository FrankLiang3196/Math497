# Non-Structured Pruning Pruning

This directory contains a pytorch implementation of the ImageNet experiments for non-structured pruning introduced in the paper 'Training Sparse Neural Networks using $\ell^1$ Regularization'.

## Dependencies
torch v1.2, torchvision v0.4.0

## Run the code using the following command

```
CUDA_VISIBLE_DEVICES=0 python imagenet.py --lr 0.1 --lam 1e-6 --momentum 9.5 --data '~/Data/ImageNet/' --save [DIRECTORY TO STORE MODELS]
```
Please specify the GPU ID and the directory of saving the ImageNet data and models.

## Descriptions
```
All the tunable hyper-parameters are included in the command line arguments. We want to focus on tuning the initial learning rate (--lr, \~0.1), regularization parameter $\lambda$ (--lam, \~1e-6, larger $\lambda$ implies sparser model) and momentum timescale (--momentum, \~10).
```
