'''
FOR GATE SOURCE CODE: .local/lib/python3.9/site-packages/fairscale/nn/moe/top2gate.py 
FOR MOE LAYER SOURCE CODE: .local/lib/python3.9/site-packages/fairscale/nn/moe/moe_layer.py
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from fairscale.nn import MOELayer, Top2Gate
import os

### environment + distributed features setup
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backend = 'nccl' if device.type == 'cuda' else 'gloo' #gloo: CPU, nccl: GPU
dist.init_process_group(backend=backend, init_method='env://', rank=0, world_size=1) # figure out what rank and world_size are?

### WHEN INITIALIZING TENSORS OR MODELS, DON'T FORGET TO USE .TO(DEVICE)

class MyMoEModel(nn.Module):
    def __init__(self, num_experts=10, hidden_size=512):
        super(MyMoEModel, self).__init__()
                
        self.gate = Top2Gate(model_dim=hidden_size, num_experts=num_experts)
        print(self.gate)
        # Change below to whatever we want our experts to be!

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        self.moe_layer = MOELayer(self.gate, self.experts)

    def forward(self, x):
        output, loss = self.moe_layer(x)
        return output,loss

model = MyMoEModel(num_experts=10, hidden_size=512).to(device) # Actual MOE model
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 30  # Ensure this is divisible by the number of experts (e.g., 10)
seq_length = 1  # Assuming one 'token' per batch for simplicity; adjust as needed for your use case
assert batch_size % len(model.moe_layer.experts) == 0, "Batch size must be a multiple of the number of experts"

for epoch in range(10):
    model.train()
    # Create a 3D input tensor [batch_size, seq_length, hidden_size]
    inputs = torch.randn(batch_size, seq_length, 512)  # Adjusted to add a 'token' dimension
    inputs = inputs.to(device)

    # This loss is from the GATE (whatever is called in top2gating())

    outputs, loss = model(inputs)
    #primary_loss = outputs.mean()
    #total_loss = primary_loss 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Primary Loss: {loss.item()}, Auxiliary Loss: {loss.item()}')
