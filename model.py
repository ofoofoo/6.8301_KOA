import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import av
import numpy as np
import torchvision
from huggingface_hub import hf_hub_download
from torch import einsum
from datasets import load_dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from datasets import load_dataset
import torch.nn.functional as F
from einops import rearrange, pack, unpack
from functools import partial
from torchvision import datasets, transforms


np.random.seed(0)

# https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce


# https://openreview.net/forum?id=TVHS5Y4dNvM

import torch.nn as nn

import timm

from urllib.request import urlopen
from PIL import Image

from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import os
import av
import numpy as np


# Setup distributed environment
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# backend = 'nccl' if device.type == 'cuda' else 'gloo'
# dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1)

import torch
import sys
sys.path.append('/home/ofoo/MoEViT/fast-kan/fastkan')

from fastkan import FastKAN as KAN

class MLP_CIFAR10(nn.Module):
    def __init__(self):
        super(MLP_CIFAR10, self).__init__()
        self.layer1 = nn.Linear(32 * 32 * 3, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MLP_CIFAR100(nn.Module):
    def __init__(self):
        super(MLP_CIFAR100, self).__init__()
        self.layer1 = nn.Linear(32 * 32 * 3, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 100)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.layer1 = nn.Linear(28*28, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class KAN_CIFAR10(nn.Module):
    def __init__(self):
        super(KAN_CIFAR10, self).__init__()
        self.layer1 = KAN([3072, 16, 10], num_grids=4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        return x

class KAN_CIFAR100(nn.Module):
    def __init__(self):
        super(KAN_CIFAR100, self).__init__()
        self.layer1 = KAN([3072, 128, 100], num_grids=3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        return x

class KAN_MNIST(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = KAN([28*28, 128, 10], num_grids=3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        return x

##################### MOE ARCHITECTURE #####################

# Define the expert model
# Define the gating model
class Gating(nn.Module):
    def __init__(self, input_dim,
                 num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()
        # Layers
        self.input_dim = input_dim
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(128, 64)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(64, num_experts)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)


        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)

        return torch.softmax(self.layer4(x), dim=1)

class MOE(nn.Module):
    def __init__(self, trained_experts, input_dim):
        super(MOE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)

        # Freezing the experts to ensure that they are not
        # learning when MoE is training.
        # Ideally, one can free them before sending the
        # experts to the MoE; in that case the following three
        # lines can be commented out.
        # for expert in self.experts:
        #     for param in expert.parameters():
        #         param.requires_grad = False

        num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        input_dim = input_dim
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x)
        disc_weights = torch.argmax(weights, dim=-1)

        # Calculate the expert outputs
        # outputs = []
        # for b in x.shape[0]:
        #     best_i_expert = disc_weights[b]
        #     subtract_dummy = 1-weights[b][best_i_expert]
        #     outputs.append((weights[b][best_i_expert]+subtract_dummy)*self.experts[best_i_expert](x))
        # output = torch.stack(outputs)
        outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2)

        # # Adjust the weights tensor shape to match the expert outputs
        # weights = weights.unsqueeze(1).expand_as(outputs)

        # # Multiply the expert outputs with the weights and
        # # sum along the third dimension
        # output = torch.sum(outputs * weights, dim=2)
        return output
    

