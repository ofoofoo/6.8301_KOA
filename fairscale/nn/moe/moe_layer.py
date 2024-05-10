from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import numpy as np
import sys
# np.set_printoptions(threshold=np.inf)

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. _Gshard: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate: gate network
        expert: expert network
        group: group to use for all-to-all communication
    """

    def __init__(self, gate: Module, experts: Union[Module, ModuleList], group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.group = group if group is not None else dist.group.WORLD
        for expert in self.experts:
            for p in experts.parameters():
                p.expert = True  # type: ignore
        self.world_size = dist.get_world_size(self.group)
        self.num_local_experts = len(self.experts)

    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        assert len(input[0].shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        assert input[0].shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"
        print("input into moe:")
        print(input[0].shape)
        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[2]
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input[0].reshape(-1, d_model)
        print('reshaped_input')
        print(reshaped_input.shape)
        self.l_aux, combine_weights, dispatch_mask = self.gate(reshaped_input)
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.float(), reshaped_input)
        print('dispacted_input before AlltoAll')
        print(dispatched_input.shape)
        dispatched_input = _AllToAll.apply(self.group, dispatched_input)
        print('dispacted_input after AlltoAll')
        print(dispatched_input.shape)
        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.world_size, self.num_local_experts, -1, d_model)
        print('dispatched_input after reshaping:')
        print(dispatched_input.shape)
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        print("chunks:")
        #print(np.shape(np.array(chunks.cpu())))
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
        #     chunk2= chunk.view(1, 32, 3, 112, 112)
        #     chunk3=dispatched_input.view(2, 32, 3, 224, 224)
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)
        expert_output = _AllToAll.apply(self.group, expert_output)
        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.world_size * self.num_local_experts, -1, d_model)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights, expert_output)
        return combined_output.reshape(input[0].shape)