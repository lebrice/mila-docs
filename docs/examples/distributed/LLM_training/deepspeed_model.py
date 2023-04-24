""" IDEA: A Transformer model that is compatible with DeepSpeed Pipeline Parallelism.
(in other words, a Transformer that has nn.Sequential-like structure.)
"""
from __future__ import annotations
from deepspeed.runtime.activation_checkpointing import checkpointing
import torch
from torch import Tensor
from transformers.models.opt import OPTModel, OPTPreTrainedModel, OPTConfig, OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTModel

from transformers.models.opt import OPTModel, OPTPreTrainedModel, OPTConfig, OPTForCausalLM
from transformers.models.opt.modeling_opt import OPTModel

# from deepspeed import PipelineModule, DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from deepspeed import PipelineModule
from deepspeed_transformer_layer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig


class PipedTransformerLayer(DeepSpeedTransformerLayer):
    """For the PipelineModule to work, we need to make sure that the forward accepts and returns
    tuples of tensors.
    """

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        hidden_states, attention_mask = inputs
        return (super().forward(hidden_states, attention_mask), attention_mask)


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
        self.pe: Tensor

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self, ntoken: int, ninp: int, nhead: int, nhid: int, nlayers: int, dropout: float = 0.5
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model=ninp, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.token_encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input: Tensor, input_mask: Tensor):
        h = self.token_encoder(input) * math.sqrt(self.ninp)
        h = self.pos_encoder(h)
        output = self.transformer_encoder(h, input_mask)
        output = self.decoder(output)
        return output


def main():
    device = torch.device("cuda")
    layer = DeepSpeedTransformerLayer(
        DeepSpeedTransformerConfig(
            batch_size=16,
            hidden_size=1024,
            intermediate_size=1024,
            heads=16,
            attn_dropout_ratio=-1,
            hidden_dropout_ratio=-1,
            num_hidden_layers=12,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            local_rank=0,
            seed=-1,
            fp16=False,
            pre_layer_norm=True,
            normalize_invertible=False,
            gelu_checkpoint=False,
            adjust_init_range=True,
            attn_dropout_checkpoint=False,
            stochastic_mode=False,
            return_tuple=False,
            training=True,
        )
    )
    layer.to(device)
    sequence_length = 128
    # TODO: Actually figure out what shapes the inputs should have.
    output = layer(
        hidden_states=torch.rand(
            [layer.config.batch_size, sequence_length, layer.config.hidden_size],
            device=device,
        ),
        attention_mask=torch.randint(
            0,
            2,
            [layer.config.batch_size, sequence_length, layer.config.hidden_size],
            device=device,
        ),
        grads=None,
    )
    print(sum(p.numel() for p in layer.parameters()), "parameters in this layer.")
    print(output.shape)
    print(output[..., 0])


if __name__ == "__main__":
    main()
