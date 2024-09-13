import torch
from torch import nn

from Experimentation.blocks.activations.gelu import PytorchGELUTanh
from Experimentation.model_configs.base import ModelConfigs


class PointWiseProjection(nn.Module):
    """
    point wise project as native from `attention is all you need`
    with slight changes in activation relu -> gelu tanh approximation
    https://arxiv.org/pdf/1706.03762
    """
    def __init__(self, config: ModelConfigs):
        super().__init__()
        self.up_projection = nn.Linear(config.d_model, config.intermediate_dim, bias=False)
        self.down_projection = nn.Linear(config.intermediate_dim, config.d_model, bias=False)
        self.act_func = PytorchGELUTanh()


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.down_projection(self.act_func(self.up_projection(input_tensor)))


class PointWiseGateProjection(nn.Module):
    """
    point wise project as native from `attention is all you need`
    with slight changes in activation relu -> gelu tanh approximation
    https://arxiv.org/pdf/1706.03762
    """

    def __init__(self, config: ModelConfigs):
        super().__init__()
        self.hidden_size = config.d_model
        self.intermediate_size = config.intermediate_dim
        self.gate_projection = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_projection = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_projection = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_func = PytorchGELUTanh()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.down_projection(self.act_func(self.gate_projection(input_tensor)) * self.up_projection(input_tensor))

