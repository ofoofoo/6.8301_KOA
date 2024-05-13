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
        super(MLP, self).__init__()
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
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(32 * 32 * 3, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(32 * 32 * 3, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class KAN_CIFAR10():
    

class MoE_Layer(nn.Module):
    def __init__(self, num_experts=4, hidden_size=32, expert_model_config=[2048, 256, 8]):
        # super(MoE_Layer, self).__init__() 
        # self.gate = Top2Gate(model_dim=hidden_size, num_experts=num_experts)
        # model = KAN(expert_model_config)  # Assuming KAN is some kind of network you have defined
        
        # self.experts = nn.ModuleList([model for _ in range(num_experts)])
        # self.moe_layer = MOELayer(self.gate, self.experts)
        model = 

    def forward(self, x):
        output = self.moe_layer(x)
        return output


class MoE_Model(nn.Module):
    def __init__(self, input_dim=32*32*3, hidden_dim=256, output_dim=10):
        super(MoE_Model, self).__init__()

        # Define parameters for MoE_Layer if different from defaults
        self.moelayer1 = MoE_Layer(num_experts=4, hidden_size=96, expert_model_config=[2048, 256, 32])
        self.moelayer2 = MoE_Layer(num_experts=4, hidden_size=96, expert_model_config=[2048, 256, 32])

        # Assuming KAN is configured properly to handle the dimensions
        self.dense1 = KAN([input_dim, hidden_dim, output_dim])

    # def forward(self, x):
    #     output = self.moelayer1(x)
    #     output = self.moelayer2(output)
    #     output = output.flatten()
    #     output = self.dense1(output)
    #     # final_output = F.softmax(output)
    #     # final_output_real = torch.argmax(final_output)
    #     return output
    def forward(self, x):
        # Assume x is [batch_size, channels, height, width]
        # Reshape x to [batch_size, tokens, features] where tokens could be height*width and features could be channels
        #x = x.view(x.size(0), -1, x.size(1))  # Reshape to [batch_size, height*width, channels]
        x = x.view(x.size(0), 32, 32*3)
        # Proceed with processing
        print(x.shape)
        print("got here")
        output = self.moelayer1(x)
        print(output)
        output = self.moelayer2(output)
        # Assuming output needs to be [batch_size, num_classes] for the final layer
        output = output.view(output.size(0), -1)  # Flatten the output for processing in a dense layer
        output = self.dense1(output)
        return F.log_softmax(output, dim=1)  # Ensure log probabilities are output



