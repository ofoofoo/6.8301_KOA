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
from soft_moe_pytorch import SoftMoE


# Setup distributed environment
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
from model import *


def train(args, model, device, train_loader, optimizer, epoch):
    
    step_loss = []
    
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.cross_entropy(output, target)
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
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)

def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Scaling KANs')

    parser.add_argument('--cifar10', action='store_true', default=False,
                        help='whether to use cifar-10 dataset')

    parser.add_argument('--cifar100', action='store_true', default=False,
                        help='whether to use cifar-100 dataset')    
                                        
    parser.add_argument('--MNIST', action='store_true', default=False,
                        help='whether to use MNIST dataset')

    parser.add_argument('--MLP', action='store_true', default=False,
                        help='whether to use vanilla MLP')

    parser.add_argument('--KAN', action='store_true', default=False,
                        help='whether to use vanilla KAN')    
                                        
    parser.add_argument('--MOEMLP', action='store_true', default=False,
                        help='whether to use MOE MLP')

    parser.add_argument('--MOEKAN', action='store_true', default=False,
                        help='whether to use MOE KAN')

    parser.add_argument('--num_experts', type=int, default=4, metavar='N',
                        help='number of MOE experts (default: 4)')

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
    
    parser.add_argument('--saverun', action='store_true', default=False,
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


    # DATA

    if args.cifar10:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Load CIFAR-10 Data
        dataset_train = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
        dataset_test = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
        if args.MLP:
            model = MLP_CIFAR10()
        elif args.KAN:
            model = KAN_CIFAR10()

        elif args.MOEMLP:
            trained_experts = [MLP_CIFAR10() for _ in range(args.num_experts)]
            print(trained_experts)
            model = MOE(trained_experts, args.num_experts)

        elif args.MOEKAN:
            trained_experts = [KAN_CIFAR10() for _ in range(args.num_experts)]
            model = MOE(trained_experts, args.num_experts)
    
    elif args.cifar100:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Load CIFAR-10 Data
        dataset_train = datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
        dataset_test = datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
        if args.MLP:
            model = MLP_CIFAR100()
        elif args.KAN:
            model = KAN_CIFAR100()

        elif args.MOEMLP:
            trained_experts = [MLP_CIFAR100() for _ in range(args.num_experts)]
            model = MOE(trained_experts, 3072)

        elif args.MOEKAN:
            trained_experts = [KAN_CIFAR100() for _ in range(args.num_experts)]
            model = MOE(trained_experts, 3072)

    
    elif args.MNIST:
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data', train=True, download=True,
                            transform=transform)
        dataset_test = datasets.MNIST('../data', train=False,
                            transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        if args.MLP:
            model = MLP_MNIST()
        elif args.KAN:
            model = KAN_MNIST()

        elif args.MOEMLP:
            trained_experts = [MLP_MNIST() for _ in range(args.num_experts)]
            model = MOE(trained_experts, 28*28)

        elif args.MOEKAN:
            trained_experts = [KAN_MNIST() for _ in range(args.num_experts)]
            model = MOE(trained_experts, 28*28)
    #Initialize the model
    
    model = model.to(device)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    
    print(f'Total parameters: {total_params}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr) #args.lr

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    training_loss = []
    test_loss_list = []
    test_accuracy_list = []
    for epoch in range(1, args.epochs + 1):
        step_loss = train(args, model, device, train_loader, optimizer, epoch)
        training_loss += step_loss
        test_loss, test_accuracy = test(model, device, test_loader)
        test_accuracy_list.append(test_accuracy)
        test_loss_list.append(test_loss)
        scheduler.step()
    
    # Save run metrics:
    # Training loss, test loss, test accuracy, total parameters

    #Save step loss
    if args.saverun:
        file_path = "/home/ofoo/MoEViT/results/"

        if args.cifar10:
            file_path += "cifar10/cifar10_"
        if args.cifar100:
            file_path += "cifar100/cifar100_"
        if args.MNIST:
            file_path += "mnist/mnist_"
        if args.MLP:
            file_path += "MLP"
        if args.KAN: 
            file_path += "KAN"
        if args.MOEMLP:
            file_path += "MOEMLP"
        if args.MOEKAN:
            file_path += "MOEKAN"
        
        torch.save(model.state_dict(), file_path + ".pth") # save model
        


    #print(training_loss)
    #print(test_loss)

if __name__ == '__main__':
    main()