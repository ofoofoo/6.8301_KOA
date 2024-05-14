import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import av
import numpy as np
import torchvision
import torchvision.transforms as transforms
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
import timm
from einops.layers.torch import Rearrange, Reduce
from urllib.request import urlopen
from PIL import Image
import os
import sys
import re

directory = '/home/ofoo/MoEViT/results/cifar10/mixers'
dir = os.listdir(directory)
import os
import numpy as np

files = os.listdir(directory)

total_params_files = [file for file in files if file.endswith('test_accuracy.npy')]

arrays = {}
for file in total_params_files:
    file_path = os.path.join(directory, file)
    array = np.load(file_path)
    pattern = r'([^/]+)_test_accuracy\.npy$'
    match = re.search(pattern, file_path)
    result = match.group(1)
    arrays[result] = array

print(arrays)

colors = {
    "S4MLPMixer": "red",
    "S8MLPMixer": "green",
    "S16MLPMixer": "orange",
    "L4MLPMixer": "blue",
    "L8MLPMixer": "purple",
    "L16MLPMixer": "gray",
}

epochs = np.arange(1, 21)
linewidth = 1
for key, value in arrays.items():
    value = value * 100.
    if 'MLPMixer' in key:
        # Plot MLPMixer models as solid lines
        if 'L4' in key:
            plt.plot(epochs, value, '--', label=key, color=colors[key], alpha = 1, linewidth = linewidth)
        else:
            plt.plot(epochs, value, '--', label=key, color=colors[key], alpha=0.5, linewidth = linewidth)

    elif 'KANMixer' in key:
        # Extract the corresponding MLPMixer color and plot as dashed
        base_model = key.replace('KANMixer', 'MLPMixer')
        if 'S4' in key:
            plt.plot(epochs, value, label=key, color=colors[base_model], alpha=1, linewidth = linewidth)
        else:
            plt.plot(epochs, value, label=key, color=colors[base_model], alpha=0.5, linewidth = linewidth)

save_directory = '/home/ofoo/MoEViT/results/cifar10/mixers'
plt.title("KANMixer/MLPMixer Top-1 Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy (%)")
plt.legend(loc='best', ncol=2)
plt.grid(axis='y')
plt.xticks(np.arange(1, 21))
plt.yticks([25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
file_path = os.path.join(save_directory, 'mixer_comparison.png')
plt.savefig(file_path)
