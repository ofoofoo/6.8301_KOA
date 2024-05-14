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
        self.layer1 = nn.Linear(32 * 32 * 3, 1024)
        self.layer2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
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
        self.layer1 = KAN([3072, 1024, 10], num_grids=4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        return x

class KAN_CIFAR100(nn.Module):
    def __init__(self):
        super(KAN_CIFAR100, self).__init__()
        self.layer1 = KAN([3072, 2048, 1024, 512, 256, 128, 100], num_grids=4)

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
        self.layer1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(512, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(128, num_experts)

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
        #print('gating')
        #print(self.gating)

        self.input_dim = input_dim

    def forward(self, x):
        # Get the weights from the gating network
        x_weights = x.view(-1, 32*32*3)

        # Get the weights from the gating network
        weights = self.gating(x_weights)
        # Calculate the expert outputs
        outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2)
        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1).expand_as(outputs)
        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        output = torch.sum(outputs * weights, dim=2)
        return output

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    print('hi')
    print(nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    ))
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def FeedForwardKAN(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    #print('hi')
    print(nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    ))
    return KAN([dim, inner_dim, dim])
    # return KAN([dim, inner_dim, dim])

def KANMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    #print("wtf")
    #print(FeedForward(num_patches, expansion_factor, dropout, chan_first))
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForwardKAN(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class CIFAR10_MLP_Mixer(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mixer = MLPMixer(
            image_size = (32, 32),
            channels = 3,
            patch_size = 16,
            dim = 512,
            depth = 4,
            num_classes = 10
        )

    def forward(self, x):
        #print(x.shape)
        output = F.log_softmax(self.mixer(x))
        return output

class CIFAR10_KAN_Mixer(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mixer = KANMixer(
            image_size = (32, 32),
            channels = 3,
            patch_size = 4,
            dim = 768,
            depth = 10,
            num_classes = 10
        )

    def forward(self, x):
        #print(x.shape)
        output = F.log_softmax(self.mixer(x))
        return output

# L16:
# image_size = (32, 32),
#             channels = 3,
#             patch_size = 16,
#             dim = 768,
#             depth = 10,
#             num_classes = 10
# L8:
# image_size = (32, 32),
#             channels = 3,
#             patch_size = 8,
#             dim = 768,
#             depth = 10,
#             num_classes = 10
# L4:
# image_size = (32, 32),
#             channels = 3,
#             patch_size = 4,
#             dim = 768,
#             depth = 10,
#             num_classes = 10
# S16:
# image_size = (32, 32),
#             channels = 3,
#             patch_size = 16,
#             dim = 512,
#             depth = 4,
#             num_classes = 10
# S8:
# image_size = (32, 32),
#             channels = 3,
#             patch_size = 8,
#             dim = 512,
#             depth = 4,
#             num_classes = 10
# S8:
# image_size = (32, 32),
#             channels = 3,
#             patch_size = 4,
#             dim = 512,
#             depth = 4,
#             num_classes = 10
