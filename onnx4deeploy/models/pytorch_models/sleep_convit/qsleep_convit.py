""" This is a ViT model fur sleep staging / sleep stage classification """

import torch
import torch.nn as nn
import math
from collections import OrderedDict
from functools import partial
import numpy as np
import brevitas.nn as qnn
import torch.nn.functional as F
from brevitas.inject.enum import FloatToIntImplType
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int32Bias,
    Int8WeightPerChannelFloat,
    Int8WeightPerTensorFloat,
    Uint8ActPerTensorFloat)

from brevitas.quant_tensor import QuantTensor

# local imports
from DeepQuant.ExportBrevitas import exportBrevitas



class Int8ActStochasticPerTensorFloat(Int8ActPerTensorFloat):
    float_to_int_impl_type=FloatToIntImplType.STOCHASTIC_ROUND  

class Int8WeightStochasticPerTensorFloat(Int8WeightPerTensorFloat):
    float_to_int_impl_type=FloatToIntImplType.STOCHASTIC_ROUND

class Int8WeightStochasticPerChannelFloat(Int8WeightPerChannelFloat):
    float_to_int_impl_type=FloatToIntImplType.STOCHASTIC_ROUND

class Int32BiasStochastic(Int32Bias):
    float_to_int_impl_type=FloatToIntImplType.STOCHASTIC_ROUND

class Uint8StochasticActPerTensorFloat(Uint8ActPerTensorFloat):
    float_to_int_impl_type=FloatToIntImplType.STOCHASTIC_ROUND

convAndLinQuantParamsNoOutputQuant = {
    "weight_bit_width": 8,
    "bias_quant": Int32Bias,
    "input_quant": Int8ActPerTensorFloat,
    # "weight_quant": Int8WeightPerChannelFloat,
    "weight_quant": Int8WeightPerChannelFloat,
    "output_quant": Int8ActPerTensorFloat,
    "return_quant_tensor": True,
}

convAndLinQuantParams = {
    "weight_bit_width": 8,
    "bias_quant": Int32Bias,
    "input_quant": Int8ActPerTensorFloat,
    "weight_quant": Int8WeightPerChannelFloat,
    "output_quant": Int8ActPerTensorFloat,
    "return_quant_tensor": True,
}

convAndLinQuantParamsNoInputQuant = {
    "weight_bit_width": 8,
    "output_bit_width": 8,
    "bias_quant": Int32Bias,
    "input_quant": None,
    "weight_quant": Int8WeightPerChannelFloat,
    "output_quant": Int8ActPerTensorFloat,
    "return_quant_tensor": True,
}

convAndLinQuantParamsNoInputNoOutputQuant = {
    "weight_bit_width": 8,
    "bias_quant": Int32Bias,
    "input_quant": None,
    # "weight_quant": Int8WeightPerChannelFloat,
    "weight_quant": Int8WeightPerChannelFloat,
    "output_quant": None,
    "return_quant_tensor": False,
}

actQuantParams = {
    "act_quant": Int8ActPerTensorFloat,
    "return_quant_tensor": True,
    "bit_width": 8
}


mhaQuantParams = {
    "in_proj_input_quant":Int8ActPerTensorFloat,
    "in_proj_weight_quant":Int8WeightPerChannelFloat,
    "in_proj_bias_quant":Int32Bias,
    "attn_output_weights_quant":Uint8ActPerTensorFloat,
    "q_scaled_quant":Int8ActPerTensorFloat,
    "k_transposed_quant":Int8ActPerTensorFloat,
    "v_quant":Int8ActPerTensorFloat,
    "out_proj_input_quant":Int8ActPerTensorFloat,
    "out_proj_weight_quant":Int8WeightPerChannelFloat,
    "out_proj_bias_quant":Int32Bias,
    "out_proj_output_quant":Int8ActPerTensorFloat,
    "return_quant_tensor":True
}

class  MLPHead(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate=0.0):
        super(MLPHead, self).__init__()
        self.ff1 = qnn.QuantLinear(dim, hidden_dim, **convAndLinQuantParams)
        self.activation = F.gelu
        #self.dropout1 = nn.Dropout(p=dropout_rate)
        self.ff2 = qnn.QuantLinear(hidden_dim, dim, **convAndLinQuantParams)
        #self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # input dim = encoder dim = 48
        x = self.ff1(x)
        x = self.activation(x)
        #x = self.dropout1(x)
        x = self.ff2(x)
        #x = self.dropout2(x)
        return x

class Encoder(nn.Module):
    """
    Transformer Encoder block with multi-head attention and feedforward network.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        seq_len: Fixed sequence length
        batch_size: Fixed batch size
        att_dropout: Attention dropout rate (ignored in deploy version)
        mlp_head_hidden_dim: Hidden dimension for MLP head
        mlp_head_dropout: MLP dropout rate (ignored in deploy version)
    """
    def __init__(self,
                 embed_dim,
                 nheads,
                 att_dropout,
                 mlp_head_hidden_dim,
                 mlp_head_dropout):
        super(Encoder, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.mha = qnn.QuantMultiheadAttention(embed_dim,
                                                nheads,
                                                dropout=att_dropout,
                                                batch_first=True,
                                                packed_in_proj=False,
                                                **mhaQuantParams)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ff = MLPHead(embed_dim,
                          hidden_dim=mlp_head_hidden_dim,
                          dropout_rate=mlp_head_dropout)
        self.rescale_residual1 = qnn.QuantIdentity(**actQuantParams)
        self.rescale_residual2 = qnn.QuantIdentity(**actQuantParams)
        # self.residual1 = qnn.QuantEltwiseAdd(**actQuantParams)
        # self.residual2 = qnn.QuantEltwiseAdd(**actQuantParams)


    def forward(self, x):
        # save x for residual
        _x = x
        x = self.ln_1(x)
        x, _ = self.mha(x, x, x, need_weights=False)
        # apply residual
        x = self.rescale_residual1(x)
        _x = self.rescale_residual1(_x)
        x = x + _x
        _x = x
        x = self.ln_2(x)
        x = self.ff(x)
        # apply residual
        x = self.rescale_residual2(x)
        _x = self.rescale_residual2(_x)
        x = x + _x
        return x

class ConvStem(nn.Module):
    def __init__(self, in_channels=1, out_channels=48, kernel_sizes=(25, 200, 100), stride=4, pool_kernel=4):
        """
        CNN branch for multi-scale feature extraction.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Total number of output channels across all branches.
            kernel_sizes (tuple): Kernel sizes for the three branches.
            stride (int): Stride for downsampling in convolutional layers.
            pool_kernel (int): Kernel size for pooling layers.
        """
        super(ConvStem, self).__init__()
        branch_out_channels = out_channels // 3

        # Branch 1: Kernel size 25
        self.branch1 = nn.Sequential(
            qnn.QuantConv2d(
                in_channels=in_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, kernel_sizes[2]),  # (height, width)
                stride=(1, stride),
                padding=(0, kernel_sizes[2] // 2),
                bias=True,
                **convAndLinQuantParamsNoOutputQuant),
            qnn.QuantReLU(bit_width=8, return_quant_tensor=True),
            nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)),
            qnn.QuantConv2d(
                in_channels=branch_out_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                bias=True,
                **convAndLinQuantParamsNoOutputQuant),
            qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        )
        
        # Branch 2: Kernel size 200
        self.branch2 = nn.Sequential(
            qnn.QuantConv2d(
                in_channels=in_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, kernel_sizes[1]),  # (height, width)
                stride=(1, stride),
                padding=(0, kernel_sizes[1] // 2),
                bias=True,
                **convAndLinQuantParamsNoOutputQuant),
            qnn.QuantReLU(bit_width=8, return_quant_tensor=True),
            nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)),
            qnn.QuantConv2d(
                in_channels=branch_out_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                bias=True,
                **convAndLinQuantParamsNoOutputQuant),
            qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        )
        
          
        # Branch 3: Kernel size 100
        self.branch3 = nn.Sequential(
            qnn.QuantConv2d(
                in_channels=in_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, kernel_sizes[2]),  # (height, width)
                stride=(1, stride),
                padding=(0, kernel_sizes[2] // 2),
                bias=True,
                **convAndLinQuantParamsNoOutputQuant),
            qnn.QuantReLU(bit_width=8, return_quant_tensor=True),
            nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)),
            qnn.QuantConv2d(
                in_channels=branch_out_channels,
                out_channels=branch_out_channels,
                kernel_size=(1, 3),
                stride=(1, 2),
                padding=(0, 1),
                bias=True,
                **convAndLinQuantParamsNoOutputQuant),
            qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        )
        
        self.cat_rescale = qnn.QuantIdentity(**actQuantParams)


    def forward(self, x):
        """
        Forward pass through dual-branch convolutional stem.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               Expected: (B, 1, 1, 3000)

        Returns:
            Concatenated features from both branches (B, model_dim, 1, num_patches)
        """
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        x1 = self.cat_rescale(x1)
        x2 = self.cat_rescale(x2)
        x3 = self.cat_rescale(x3)
        x12 = torch.cat((x1, x2), dim=1)
        x123 = torch.cat((x12, x3), dim=1)
        return x123

class QSleepConViT(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            config:dict):
            # img_size=224,
            # patch_size=16,
            # in_chans=3,
            # num_classes=1000,
            # embed_dim=768,
            # depth=12,
            # num_heads=12,
            # mlp_ratio=4.0,
            # qkv_bias=True,
            # qk_scale=None,
            # representation_size=None,
            # drop_rate=0.0,
            # attn_drop_rate=0.0,
            # drop_path_rate=0.0,
            # norm_layer=None):
        super().__init__()
        self.num_heads = config.get("num_heads", 8)
        self.model_dim = config.get("model_dim", 48)
        self.num_patches = config.get("num_patches", 94)
        self.num_classes = config.get("num_classes", 4)
        self.batch_size = config.get("batch_size", 1)
        seq_len = config.get("seq_len", 95)  # num_patches + 1 (for CLS token)

        self.conv_stem = ConvStem(in_channels=1,
                                  out_channels=self.model_dim,
                                  kernel_sizes=(25, 200, 100),
                                  stride=4)

        num_patches = config["num_patches"]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.model_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.model_dim))
        
        # CLS token selector (fixed one-hot vector for ONNX-friendly extraction)
        self.cls_selector = nn.Parameter(torch.zeros(1, self.num_patches + 1), requires_grad=False)
        self.cls_selector.data[0, 0] = 1.0  # Select only the first token (CLS)

        self.pos_drop = nn.Dropout(p=config["attention_dropout"])

        self.qaddpos = qnn.QuantIdentity(**actQuantParams)

        self.encoder = Encoder(
            self.model_dim,
            config["num_heads"],
            config["attention_dropout"],
            config["mlp_head_hidden_dim"],
            config["encoder_ff_dropout"])
        self.norm = nn.LayerNorm(self.model_dim, eps =1e-6)
        self.rescale_norm = qnn.QuantIdentity(**actQuantParams)

        self.head = qnn.QuantLinear(self.model_dim, self.num_classes, **convAndLinQuantParams)

    def forward(self, x):
        x = self.conv_stem(x)

        x = x.reshape(self.batch_size, self.model_dim, self.num_patches).permute(0, 2, 1)

        cls_tokens = self.cls_token.expand(self.batch_size, -1, -1) 
        cls_tokens = self.qaddpos(cls_tokens)
        x = self.qaddpos(x)
        x_cls = torch.cat((cls_tokens, x), dim=1)
        pos = self.qaddpos(self.pos_embed)
        x_pos = x_cls + pos
        x = self.encoder(x_pos)
        x = self.norm(x)
        x = torch.matmul(self.cls_selector, x)
        x = x.squeeze(1)  # [B, 1, 48] -> [B, 48]
        x = self.rescale_norm(x)
        print(F"hello")
        x = self.head(x)
        print(F"hello2222")

        return x
