import argparse
import os
import random
import shutil
import time
import warnings
import sys
import torch
from collections import OrderedDict


# model = torch.load('./results/vgg16_sparse_model.dat')
# print('Total params: %d' % (sum(p.numel() for p in model.parameters())))

# print('Params sparsity: %.4f' % (sum(torch.nonzero(x).size()[0]
#                                      for x in list(model.parameters())) / sum(p.numel() for p in model.parameters())))

# print('Kernel number: %d' % (
#     sum(p.shape[0] * p.shape[1] for p in model.parameters() if len(p.shape) == 4)))

# print('Kernel sparsity: %d' % (sum(len(torch.norm(p, p=2, dim=(2, 3)).nonzero())
#                                    for p in model.parameters() if len(p.shape) == 4)))

# print('Channel number: %d' % (sum(len(torch.norm(p, p=2, dim=(0, 2, 3)))
#                                   for p in model.parameters() if len(p.shape) == 4)))

# print('Channel sparsity: %.4f' % (sum(len(torch.norm(p, p=2, dim=(0, 2, 3)).nonzero()) for p in model.parameters() if len(p.shape) == 4) /
#                                   sum(len(torch.norm(p, p=2, dim=(0, 2, 3))) for p in model.parameters() if len(p.shape) == 4)))


model = torch.load('./results/resnet56_sparse_model_cosine_7en7.dat')
print('Total params: %d' % (sum(p.numel() for p in model.values())))

print('Params sparsity: %.4f' % (sum(torch.nonzero(x).size()[0]
                                     for x in list(model.values())) / sum(p.numel() for p in model.values())))

print('Kernel number: %d' % (
    sum(p.shape[0] * p.shape[1] for p in model.values() if len(p.shape) == 4)))

print('Sparse kernel: %d' % (sum(len(torch.norm(p, p=2, dim=(2, 3)).nonzero())
                                 for p in model.values() if len(p.shape) == 4)))

print('Kernel sparsity: %.4f' % (sum(len(torch.norm(p, p=2, dim=(2, 3)).nonzero())
                                     for p in model.values() if len(p.shape) == 4) / sum(p.shape[0] * p.shape[1] for p in model.values() if len(p.shape) == 4)))
