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
from model import MoEModel, sample_frame_indices, read_video_pyav, MLPMixer, CIFAR10_MLP_Mixer, ConvMixer
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

#model = MoEModel(num_experts=8, hidden_size=32).to(device)
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary


def train(args, model, device, train_loader, optimizer, epoch):
    
    step_loss = []
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        step_loss.append(loss.item())
        
    return step_loss

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 MLP Mixer')
    
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 1.0)')
    
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size,
                    'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load CIFAR-10 Data
    dataset_train = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    dataset_test = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    #Initialize the model
    
    model = ConvMixer(128, 8, kernel_size=8, patch_size=1, n_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(total_params)

      

    optimizer = optim.Adam(model.parameters(), lr=args.lr) #args.lr

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    training_loss = []
    test_loss_list = []
    for epoch in range(1, args.epochs + 1):
        step_loss = train(args, model, device, train_loader, optimizer, epoch)
        training_loss += step_loss
        test_loss = test(model, device, test_loader)
        test_loss_list.append(test_loss)
        scheduler.step()
        
    #Save step loss
    #print(training_loss)
    #print(test_loss)

if __name__ == '__main__':
    main()

# input = torch.rand(32, 32, 32)

# ####### testing data shape
# print(model(input))
# # video clip consists of 300 frames (10 seconds at 30 FPS)
# file_path = '/home/ofoo/MoEViT/kinetics-dataset/k400/train/-_1WRslPhMo_000173_000183.mp4'
# container = av.open(file_path)

# image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")



# # sample 32 frames
# indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
# video = read_video_pyav(container=container, indices=indices)
# inputs = image_processor(list(video), return_tensors="pt") 
# inputs = inputs['pixel_values']

# print(f'Input to ViViT: {inputs.shape}') # (batch_size, num_frames, channels, height, width) 

# batch_size, num_frames, channels, height, width = inputs.shape

# # model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
# # print(model(inputs))

# num_tokens = 49  # For instance, if each 224x224 frame is divided into 49 patches.
# model_dim = 3072  # Hypothetical feature dimension per patch.

# inputs_vivit = inputs
# inputs_moe = inputs.view(-1, num_tokens, model_dim).to(device)  # Reshape from [1, 32, 3, 224, 224] to [batch_size * num_frames, num_tokens, model_dim]
# #inputs = inputs.view(batch_size * num_frames, channels, height, width)


# print(f'Kinetics 400 Video shape: {video.shape}')


# #inputs=inputs.reshape(inputs.shape[1], inputs.shape[3], -1).to(device) # figure out how to undo this, need to fix this INSIDE of the Moe layer (fairscale) source code?
# #print("Reshaped inputs shape: ", inputs.shape)
# #try random size

# #what should the input to the model be?
# output= model(inputs_moe)
# print(output)
