import torch
from torch import nn

from Experimentation.model_configs.base import ModelConfigs


class IOEmbedding(nn.Module):
    """
    This is an Input output embedding space used inside the transformers and other architectures

    :param
    d_model -> Models dimension
    vocabulary_size -> model vocab size
    padding_idx -> padding index
    """

    def __int__(self, config: ModelConfigs):
        self.d_model = config.d_model
        self.vocabulary_size = config.vocabulary_size

        self.embeddings = nn.Embedding(num_embeddings=config.vocabulary_size,
                                       embedding_dim=config.d_model,
                                       padding_idx=config.padding_id)

        # positional_tensor = [1, 2, 3, 4, 5, ..., n -1]
        # positional_tensor.expand -> creating a view of previous tensor [[1, 2, 3, .... , n-1]]
        self.register_buffer("positional_id", torch.arange(self.max_positions).expand(1, -1))

    def forward(self, input_ids: torch.tensor):
        """embedding forward pass"""
        """
        input_ids -> [3, 4, 2, 4]
        
        # sample:
        model_dim -> 3
        model_embedding = [[1, 2, 3]
                            [0.2, 0.3, -0.2, 0.3],
                            [0.12, 0.323, -0.22, 0.3],
                            [0.2, 0.3, -0.2, 0.3]
                            ]
                            
        
        embedding = (model_embedding + positional_embedding 
        """
        model_embedding = self.embeddings(input_ids)
        position_embedding = self.position_id[:, :input_ids.size()[0]]
        return model_embedding + position_embedding
