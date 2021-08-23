#!/usr/bin/python
# Library file for first training on cifar-10
# Created by Feyzullah KARA

# Main libraries for every learning algorithm
import sys, os
import pickle
import numpy as np
from tqdm import tqdm

# For the plotting any image or visualization of analysis
import matplotlib.pyplot as plt

# The needed pyTorch libraries and models
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import Sigmoid, ReLU, Conv2d, MaxPool2d, Dropout, Module, Sequential, CrossEntropyLoss, Linear
from torch.nn import AdaptiveAvgPool2d, BatchNorm1d
from torch.optim import Adam, SGD
from torch.nn.functional import relu, sigmoid, rrelu, max_pool2d, dropout, interpolate, celu, gelu, softmax
from torch import flatten
import torch.cuda
import torch.hub as hub

