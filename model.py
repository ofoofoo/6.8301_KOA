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
import torch
import torchvision
from transformers import VivitImageProcessor, VivitForVideoClassification
from huggingface_hub import hf_hub_download

from datasets import load_dataset
from torch.utils.data import DataLoader


from datasets import load_dataset

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


# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path = hf_hub_download(
    repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
)
file_path = '/home/ofoo/MoEViT/kinetics-dataset/k400/train/-_1WRslPhMo_000173_000183.mp4'
container = av.open(file_path)

# sample 32 frames
indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container=container, indices=indices)
print("testaiosjglkagjakgjaslASLKGJALKGJASLKGJASLKGJASKL GJSALKG ALSG JASLKG JASILG JAKLSG JLSAG JAKLSG JAKLS JGILSD ")
print(video)
print(type(video))
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

print(video)
inputs = image_processor(list(video), return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# model predicts one of the 400 Kinetics-400 classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])

class MoEModel(nn.Module):
    def __init__(self, num_experts=10, hidden_size=512):
        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
        print(model)
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
