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
from model import *

# ### WHEN INITIALIZING TENSORS OR MODELS, DON'T FORGET TO USE .TO(DEVICE)

# def main():

#     ### environment + distributed features setup

#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     backend = 'nccl' if device.type == 'cuda' else 'gloo' #gloo: CPU, nccl: GPU
#     dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1) # figure out what rank and world_size are?

#     # DATA 
#     transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     # Load CIFAR-100 training data
#     trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
#                                             download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
#                                             shuffle=True, num_workers=2)

#     # Load CIFAR-100 testing data
#     testset = torchvision.datasets.CIFAR100(root='./data', train=False,
#                                         download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=64,
#                                             shuffle=False, num_workers=2)

#     classes = trainset.classes

#     model = MoEModel(num_experts=10, hidden_size=512).to(device) # Actual MOE model
#     print(model)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     batch_size = 100  # Ensure this is divisible by the number of experts (e.g., 10)
#     seq_length = 10  # Assuming one 'token' per batch for simplicity; adjust as needed for your use case
#     assert batch_size % len(model.moe_layer.experts) == 0, "Batch size must be a multiple of the number of experts"

#     for epoch in range(10):
#         model.train()
#         # Create a 3D input tensor [batch_size, seq_length, hidden_size]
#         inputs = torch.randn(batch_size, seq_length, 512)  # Adjusted to add a 'token' dimension
#         inputs = inputs.to(device)

#         # This loss is from the GATE (whatever is called in top2gating())

#         outputs, loss = model(inputs)
#         #primary_loss = outputs.mean()
#         #total_loss = primary_loss 
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(f'Epoch {epoch+1}, Primary Loss: {loss.item()}, Auxiliary Loss: {loss.item()}')

# if __name__ == '__main__':
#     main()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from fairscale.nn import MOELayer, Top2Gate
# import os
# import torch.distributed as dist
# from torchvision import datasets, transforms


# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# backend = 'nccl' if device.type == 'cuda' else 'gloo' #gloo: CPU, nccl: GPU
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1) # figure out what rank and world_size are?


# gates1 = torch.randn(64, 1000, 1, device='cuda')  # Simulate batch size of 64 and 1000 experts
# locations1_sc = torch.randn(64, 1, 1000, device='cuda')  # Simulate 1000 location classes

# # Perform the problematic operation
# result = gates1.unsqueeze(2) * locations1_sc.unsqueeze(1)

# print("Result shape:", result.shape)
# print("Memory status:", torch.cuda.memory_summary())
# Data transformation and loading
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=66,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR100(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=66,
#                                          shuffle=False, num_workers=2)

# transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#         ])
    
# #Load MNIST data
# dataset1 = datasets.MNIST('../data', train=True, download=True,
#                     transform=transform)
# dataset2 = datasets.MNIST('../data', train=False,
#                     transform=transform)

# train_loader = torch.utils.data.DataLoader(dataset1, batch_size = 64)
# test_loader = torch.utils.data.DataLoader(dataset2, batch_size = 64)

# # Define the MoE Model
# class MoEModel(nn.Module):
#     def __init__(self, num_experts=3, hidden_size=4, num_classes=100):
#         super(MoEModel, self).__init__()
#         self.num_experts = num_experts
#         # self.conv1 = nn.Conv2d(3, 6, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # self.fc1 = nn.Linear(16 * 5 * 5, hidden_size)
#         self.moe_layer = MOELayer(
#             gate=Top2Gate(model_dim=hidden_size, num_experts=num_experts),
#             experts=nn.ModuleList([nn.Sequential(
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(hidden_size, num_classes)
#             ) for _ in range(num_experts)]),
#         )
        
#     def forward(self, x):
#         # print("Shape of input to MoE layer:", x.shape)  # This should be [batch_size, num_experts, hidden_size]
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         # x = torch.flatten(x, 1)
#         # x = F.relu(self.fc1(x))
#         # # Artificially expand the token dimension to be a multiple of the number of experts
#         # x = x.unsqueeze(1).expand(-1, self.num_experts, -1)
#         # print(x)
#         # print(x.shape)
#         # print("it breaks here")
#         print(x.shape)
#         x, _ = self.moe_layer(x)
#         # Inside your MoEModel's forward method

#         return x.mean(dim=1)  # Aggregate over the token dimension
# model = MoEModel().cuda()

# # Define Loss function and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # Training loop
# for epoch in range(10):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data[0].cuda(), data[1].cuda()

#         print(inputs.shape)
#         print(inputs[0].shape[0])
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i % 200 == 199:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
#             running_loss = 0.0

# print('Finished Training')

# # Test the model
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].cuda(), data[1].cuda()
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# # Define the Mixture of Experts model
# class MyMoEModel(nn.Module):
#     def __init__(self, num_experts=1, hidden_size=256, num_classes=10):
#         super(MyMoEModel, self).__init__()
#         self.moe_layer = MOELayer(
#             gate=Top2Gate(model_dim=hidden_size, num_experts=num_experts),
#             experts=nn.ModuleList([nn.Sequential(
#                 nn.Linear(hidden_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Linear(hidden_size, num_classes)
#             ) for _ in range(num_experts)])
#         )
        
#     def forward(self, x):
#         print(x.shape)
#         x, _ = self.moe_layer(x)  # Pass the input through the MoE layer
#         return x.mean(dim=1)  # Aggregate the outputs

# # Initialize the model and move it to GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MyMoEModel().to(device)
# print("Memory status:", torch.cuda.memory_summary())

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# # Training loop
# def train_model():
#     model.train()
#     for epoch in range(10):
#         running_loss = 0.0
#         for images, labels in train_loader:
#             # Flatten and prepare inputs
#             inputs = images.view(images.shape[0], -1).unsqueeze(2).to(device)
#             labels = labels.to(device)
            
#             # Zero the parameter gradients
#             optimizer.zero_grad()
            
#             # Forward + backward + optimize
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#         print(f'Epoch {epoch+1} loss: {running_loss/len(train_loader)}')
#     print('Finished Training')

# # Function to test the model
# def test_model():
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             inputs = images.view(images.shape[0], -1).unsqueeze(2).to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# # Run training and testing
# train_model()
# test_model()

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

# Setup distributed environment
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = 'nccl' if device.type == 'cuda' else 'gloo'
dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1)

class MyMoEModel(nn.Module):
    def __init__(self, num_experts=8, model_dim=784, output_dim=10):
        super(MyMoEModel, self).__init__()
        self.gate = Top2Gate(model_dim=model_dim, num_experts=num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            ) for _ in range(num_experts)
        ])
        self.moe_layer = MOELayer(self.gate, self.experts)

    def forward(self, x):
        print(x.shape)
        output, loss = self.moe_layer(x)
        return output, loss


model = MoEModel(num_experts=8, hidden_size=512).to(device)
print(model)
# # Load MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# # Model
# model = MyMoEModel(num_experts=8, model_dim=784, output_dim=10).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# # Training loop
# for epoch in range(10):
#     model.train()
#     for images, labels in trainloader:
#         images = images.view(-1, 784).to(device)  # Flatten MNIST images
#         images = images.unsqueeze(1)
#         labels = labels.to(device)

#         outputs, loss_aux = model(images)
#         loss_primary = criterion(outputs, labels)
#         total_loss = loss_primary + loss_aux  # Combine primary and auxiliary losses

#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch+1}, Loss: {total_loss.item()}')

