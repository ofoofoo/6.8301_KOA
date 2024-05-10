import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from fairscale.nn import MOELayer, Top2Gate
# Load model directly
# Load model directly
from transformers import AutoTokenizer, AutoModelForVideoClassification
import av
import numpy as np
import torchvision
from transformers import VivitImageProcessor, VivitForVideoClassification
from transformers import VivitConfig, VivitModel
from huggingface_hub import hf_hub_download

from datasets import load_dataset
from torch.utils.data import DataLoader

# If the dataset is gated/private, make sure you have run huggingface-cli login
import matplotlib.pyplot as plt
from datasets import load_dataset
dataset = load_dataset("theodor1289/imagenet-1k_tiny")
import torch.nn.functional as F
from functools import partial

#dataset = load_dataset("AlexFierro9/Kinetics400")

# dataset = load_dataset("kiyoonkim/kinetics-400-splits")
# train_dataset = load_dataset("kiyoonkim/kinetics-400-splits", split="train")
# valid_dataset = load_dataset("kiyoonkim/kinetics-400-splits", split="validation")
# test_dataset  = load_dataset("kiyoonkim/kinetics-400-splits", split="test")
# print(train_dataset)
# for example in train_dataset:
#     print(example)
# #dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# # Iterate over the data
# for video, audio, label in dataloader:
#     print(video.shape, audio.shape, label)

#dataset = load_dataset("AlexFierro9/Kinetics400")

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
import timm

from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from fairscale.nn import MOELayer, Top2Gate
import os
import av
from transformers import VivitImageProcessor, VivitForVideoClassification
import numpy as np
from transformers import VivitConfig, VivitModel


# If the dataset is gated/private, make sure you have run huggingface-cli login




# img = Image.open(urlopen(
#     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
# ))

# Setup distributed environment
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
backend = 'nccl' if device.type == 'cuda' else 'gloo'
dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1)

import torch
from soft_moe_pytorch import SoftMoE

moe = SoftMoE(
    dim = 512,         # model dimensions
    seq_len = 1024,    # max sequence length (will automatically calculate number of slots as seq_len // num_experts) - you can also set num_slots directly
    num_experts = 4    # number of experts - (they suggest number of experts should be high enough that each of them get only 1 slot. wonder if that is the weakness of the paper?)
)

x = torch.randn(1, 1024, 512)

print('heasjfka')
out = moe(x) + x # (1, 1024, 512) - add in a transformer in place of a feedforward at a certain layer (here showing the residual too)
print(out)
class MoEModel(nn.Module):
    def __init__(self, num_experts=10, hidden_size=512):
        model = timm.create_model('convmixer_768_32.in1k', pretrained=True)
        # configuration = VivitConfig(image_size=112)
        # model = VivitModel(configuration)
        super(MoEModel, self).__init__() 
        self.gate = Top2Gate(model_dim=hidden_size, num_experts=num_experts)
        
        # Change below to whatever we want our experts to be!
        # self.experts = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Linear(hidden_size, hidden_size)
        #     ) for _ in range(num_experts)
        # ])
        self.experts = nn.ModuleList([
            model for _ in range(num_experts)
        ])
        
        self.moe_layer = MOELayer(self.gate, self.experts)

    def forward(self, x):
        output, loss = self.moe_layer(x)
        return output,loss

print(dataset)

print(dataset['train'].features["image"])
print(dataset['train'][2])

model = MoEModel(num_experts=10, hidden_size=224)
img = dataset['train'][4]['image']
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
print(transforms(img).shape)
output = model(transforms(img))
print(output)

# model = timm.create_model('convmixer_768_32.in1k', pretrained=True)
# model = model.eval()

# # print(dataset['train']['features'])
# count=0
# num=0
# for i in range(100):
#     print(i)

    
#     img = dataset['train'][i]['image']
#     if(img.mode!='RGB'):
#         continue
#     num+=1
#     data_config = timm.data.resolve_model_data_config(model)
#     transforms = timm.data.create_transform(**data_config, is_training=False)
#     #print(transforms(img).shape)
#     output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1
#     _, top_1 = torch.topk(output.softmax(dim=1) * 100, k = 1)
    
    
#     if dataset['train'][i]['label']==top_1:
#         count=count+1
#     else:
#         print(dataset['train'][i]['label'], top_1)
# print(count / num)

# print(output)
# print(top5_probabilities)
# print(top5_class_indices)
#net = ConvMixer(128, 8, kernel_size=8, patch_size=1, n_classes=10)

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
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
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
            patch_size = 8,
            dim = 512,
            depth = 12,
            num_classes = 10
        )

    def forward(self, x):
        output = F.log_softmax(self.mixer(x))
        return output

class MoEModel(nn.Module):
    def __init__(self, num_experts=10, hidden_size=512):
        model = timm.create_model('convmixer_768_32.in1k', pretrained=True)
        # configuration = VivitConfig(image_size=112)
        # model = VivitModel(configuration)
        super(MoEModel, self).__init__() 
        self.gate = Top2Gate(model_dim=hidden_size, num_experts=num_experts)
        
        # Change below to whatever we want our experts to be!
        # self.experts = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(hidden_size, hidden_size),
        #         nn.ReLU(),
        #         nn.Linear(hidden_size, hidden_size)
        #     ) for _ in range(num_experts)
        # ])
        self.experts = nn.ModuleList([
            model for _ in range(num_experts)
        ])
        
        self.moe_layer = MOELayer(self.gate, self.experts)

    def forward(self, x):
        output, loss = self.moe_layer(x)
        return output,loss


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# # video clip consists of 300 frames (10 seconds at 30 FPS)
# file_path = hf_hub_download(
#     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
# )
# file_path = '/home/ofoo/MoEViT/kinetics-dataset/k400/train/-_1WRslPhMo_000173_000183.mp4'
# container = av.open(file_path)

# # sample 32 frames
# indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
# video = read_video_pyav(container=container, indices=indices)
# print("testaiosjglkagjakgjaslASLKGJALKGJASLKGJASLKGJASKL GJSALKG ALSG JASLKG JASILG JAKLSG JLSAG JAKLSG JAKLS JGILSD ")
# print(video.shape)
# print(type(video)) 
# image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
# #model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
# model = MoEModel(num_experts=10, hidden_size=512)
# inputs = image_processor(list(video), return_tensors="pt")
# inputs = inputs['pixel_values']
# import json
# # Example data
# label = "LABEL12"  # This should be the output label for the video

# # Create a dictionary to store your data
# video_entry = {
#     "input": inputs,
#     "output": label
# }


# # File path to your .jsonl file
# file_path = 'your_dataset.jsonl'


# with torch.no_grad():
#     outputs = model(inputs)
#     logits = outputs.logits

# # model predicts one of the 400 Kinetics-400 classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])

