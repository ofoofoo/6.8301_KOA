# '''
# FOR GATE SOURCE CODE: .local/lib/python3.9/site-packages/fairscale/nn/moe/top2gate.py 
# FOR MOE LAYER SOURCE CODE: .local/lib/python3.9/site-packages/fairscale/nn/moe/moe_layer.py
# '''

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.distributed as dist
# from fairscale.nn import MOELayer, Top2Gate
# import os

'''
FOR GATE SOURCE CODE: .local/lib/python3.9/site-packages/fairscale/nn/moe/top2gate.py 
FOR MOE LAYER SOURCE CODE: .local/lib/python3.9/site-packages/fairscale/nn/moe/moe_layer.py
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from fairscale.nn import MOELayer, Top2Gate
import os
import av
from model import MoEModel, sample_frame_indices, read_video_pyav
from transformers import VivitImageProcessor, VivitForVideoClassification
import numpy as np
from transformers import VivitConfig, VivitModel

# Setup distributed environment
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
backend = 'nccl' if device.type == 'cuda' else 'gloo'
dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1)


###### Model - MOE we will be using 

model = MoEModel(num_experts=8, hidden_size=3072).to(device)
print(model)

####### testing data shape

# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path = '/home/ofoo/MoEViT/kinetics-dataset/k400/train/-_1WRslPhMo_000173_000183.mp4'
container = av.open(file_path)

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")



# sample 32 frames
indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container=container, indices=indices)
inputs = image_processor(list(video), return_tensors="pt") 
inputs = inputs['pixel_values']

print(f'Input to ViViT: {inputs.shape}') # (batch_size, num_frames, channels, height, width) 

batch_size, num_frames, channels, height, width = inputs.shape

# model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
# print(model(inputs))

num_tokens = 49  # For instance, if each 224x224 frame is divided into 49 patches.
model_dim = 3072  # Hypothetical feature dimension per patch.

inputs_vivit = inputs
inputs_moe = inputs.view(-1, num_tokens, model_dim).to(device)  # Reshape from [1, 32, 3, 224, 224] to [batch_size * num_frames, num_tokens, model_dim]
#inputs = inputs.view(batch_size * num_frames, channels, height, width)


print(f'Kinetics 400 Video shape: {video.shape}')


#inputs=inputs.reshape(inputs.shape[1], inputs.shape[3], -1).to(device) # figure out how to undo this, need to fix this INSIDE of the Moe layer (fairscale) source code?
#print("Reshaped inputs shape: ", inputs.shape)
#try random size

#what should the input to the model be?
output= model(inputs_moe)
print(output)
