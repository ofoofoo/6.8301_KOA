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

# Setup distributed environment
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = 'nccl' if device.type == 'cuda' else 'gloo'
dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1)


###### Model - MOE we will be using 
model = MoEModel(num_experts=8, hidden_size=512).to(device)

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

print(f'Kinetics 400 Video shape: {video.shape}')


#what should the input to the model be?
output= model(inputs)
print(output)
