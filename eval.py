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

directory = '/home/ofoo/MoEViT/results/cifar10/mixers'
dir = os.listdir(directory)
import os
import numpy as np

# Directory containing the .npy files

# List all files in the directory
files = os.listdir(directory)

# Filter for files that end with 'total_params.npy'
total_params_files = [file for file in files if file.endswith('test_accuracy.npy')]

# Initialize a list to hold the arrays
arrays = []

# Load each array and append to the list
for file in total_params_files:
    file_path = os.path.join(directory, file)
    print(file_path)
    array = np.load(file_path)
    arrays.append(array)

# Now arrays contains all the numpy arrays from the files ending with 'total_params.npy'
print(arrays)

for index, array in enumerate(arrays):
    plt.plot(array, label=f'{index}')

save_directory = '/home/ofoo/MoEViT/results/cifar10/mixers'
plt.legend()
file_path = os.path.join(save_directory, 'plot_bob.png')
plt.savefig(file_path)
