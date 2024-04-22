import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from fairscale.nn import MOELayer, Top2Gate



class MoEModel(nn.Module):
    def __init__(self, num_experts=10, hidden_size=512):
        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

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
            resnet18 for _ in range(num_experts)
        ])
        
        self.moe_layer = MOELayer(self.gate, self.experts)

    def forward(self, x):
        output, loss = self.moe_layer(x)
        return output,loss
