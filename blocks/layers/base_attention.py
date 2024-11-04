from torch import nn
from model_configs.base import ModelConfigs


class BaseAttention(nn.Module):
    """
    Base Attention module template class
    """
    def __init__(self, config: ModelConfigs):
        super().__init__()
        """
        In transformers implementation we will avoid usage of bias.
        https://ai.stackexchange.com/questions/40252/why-are-biases-typically-not-used-in-attention-mechanism
        """
        assert self.head_dim is not None

        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.attention_dim = self.num_heads * self.head_dim

        self.query_projection = nn.Linear(self.d_model, self.attention_dim, bias=False)
        self.key_projection = nn.Linear(self.d_model, self.attention_dim, bias=False)
        self.value_projection = nn.Linear(self.d_model, self.attention_dim, bias=False)

        self.output_projection = nn.Linear(self.attention_dim, self.d_model, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)
