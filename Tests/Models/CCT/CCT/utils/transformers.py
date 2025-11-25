import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Identity, LayerNorm, Linear, Module, ModuleList, Parameter, init

from .stochastic_depth import DropPath

class AttentionwLora(Module):
    """
    Attention module with explicit Q, K, V branches for ONNX export.
    Modified to use LoRA for fine-tuning.
    """
    
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = Linear(dim, dim, bias=False)
        self.k_proj = Linear(dim, dim, bias=False)
        self.v_proj = Linear(dim, dim, bias=False)
        
    
        self.q_proj.weight.requires_grad = False
        self.k_proj.weight.requires_grad = False
        self.v_proj.weight.requires_grad = False
        
    
        self.lora_r = 4  
        self.lora_alpha = 16  
        self.scaling = self.lora_alpha / self.lora_r
        
        
        self.lora_q_A = nn.Parameter(torch.zeros(self.lora_r, dim))
        self.lora_q_B = nn.Parameter(torch.zeros(dim, self.lora_r))
        self.lora_k_A = nn.Parameter(torch.zeros(self.lora_r, dim))
        self.lora_k_B = nn.Parameter(torch.zeros(dim, self.lora_r))
        self.lora_v_A = nn.Parameter(torch.zeros(self.lora_r, dim))
        self.lora_v_B = nn.Parameter(torch.zeros(dim, self.lora_r))
        
        
        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5.0))
        nn.init.normal_(self.lora_q_B, mean=0.0, std=0.001)
        
        nn.init.kaiming_uniform_(self.lora_k_A, a=math.sqrt(5.01)) 
        nn.init.normal_(self.lora_k_B, mean=0.0, std=0.00105)  
        
        nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(4.99))  
        nn.init.normal_(self.lora_v_B, mean=0.0, std=0.00095)  
        
        self.attn_drop = Dropout(attention_dropout)
        
      
        self.proj = Linear(dim, dim)
        self.proj.weight.requires_grad = False
        
       
        self.lora_proj_A = nn.Parameter(torch.zeros(self.lora_r, dim))
        self.lora_proj_B = nn.Parameter(torch.zeros(dim, self.lora_r))
        nn.init.kaiming_uniform_(self.lora_proj_A, a=math.sqrt(5.02))  
        nn.init.normal_(self.lora_proj_B, mean=0.0, std=0.00098) 
        
        self.proj_drop = Dropout(projection_dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
     
        q_base = self.q_proj(x)
        q_lora = (x @ self.lora_q_A.t() @ self.lora_q_B.t()) 
        q = (q_base + q_lora).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
     
        k_base = self.k_proj(x)
        k_lora = (x @ self.lora_k_A.t() @ self.lora_k_B.t()) 
        k = (k_base + k_lora).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
       
        v_base = self.v_proj(x)
        v_lora = (x @ self.lora_v_A.t() @ self.lora_v_B.t()) 
        v = (v_base + v_lora).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
       
        attn = (q @ k.transpose(-2, -1)) 
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
       
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
       
        proj_base = self.proj(x)
        proj_lora = (x @ self.lora_proj_A.t() @ self.lora_proj_B.t()) 
        x = proj_base + proj_lora
        
        x = self.proj_drop(x)
        return x

class Attention(Module):
    """
    Attention module with explicit Q, K, V branches for ONNX export.
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

       
        self.q_proj = Linear(dim, dim, bias=False)
        self.k_proj = Linear(dim, dim, bias=False)
        self.v_proj = Linear(dim, dim, bias=False)

        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape  
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = F.softmax(attn, dim=-1)  
        attn = self.attn_drop(attn)  

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
      
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MaskedAttention(Module):

    def __init__(self, dim, num_heads = 8, attention_dropout = 0.1, projection_dropout = 0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = Linear(dim, dim * 3, bias = False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x, mask = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            assert mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn.masked_fill_(~mask, mask_value)

        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MaskedAttentionWithLoRA(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, lora_rank=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        
        self.lora_qkv_A = nn.Parameter(torch.zeros(lora_rank, dim))
        self.lora_qkv_B = nn.Parameter(torch.zeros(dim * 3, lora_rank))
        
        nn.init.kaiming_uniform_(self.lora_qkv_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_qkv_B)
        

        self.lora_alpha = 2 * lora_rank
        self.lora_scaling = self.lora_alpha / lora_rank
        
        self.attn_drop = Dropout(attention_dropout)
        
        self.proj = Linear(dim, dim)
        
        self.lora_proj_A = nn.Parameter(torch.zeros(lora_rank, dim))
        self.lora_proj_B = nn.Parameter(torch.zeros(dim, lora_rank))
        
        nn.init.kaiming_uniform_(self.lora_proj_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_proj_B)
        
        self.proj_drop = Dropout(projection_dropout)
        
        self.lora_dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        

        qkv_original = self.qkv(x)
        
        lora_input = self.lora_dropout(x)
        lora_qkv = lora_input @ self.lora_qkv_A.t() @ self.lora_qkv_B.t() * self.lora_scaling
        
        qkv_combined = qkv_original + lora_qkv
        
        qkv = qkv_combined.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            assert mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn.masked_fill_(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        proj_original = self.proj(x)
        
        lora_proj_input = self.lora_dropout(x)
        lora_proj = lora_proj_input @ self.lora_proj_A.t() @ self.lora_proj_B.t() * self.lora_scaling
        
        proj_combined = proj_original + lora_proj
        
        x = self.proj_drop(proj_combined)
        return x


class LinearwLora(Module):
    """Linear layer with LoRA, rank=8"""
    
    def __init__(self, in_features, out_features, bias=True):
        super(LinearwLora, self).__init__()
        
        # Original linear layer
        self.linear = Linear(in_features, out_features, bias=bias)
        # requires_grad=False to avoid training the original weights
        self.linear.weight.requires_grad = False
        
        # LoRA parameters with rank=8
        self.lora_A = nn.Parameter(torch.randn(4, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, 4))
        self.scaling = 4.0  # alpha/rank = 32/8 = 4
    
    def forward(self, x):
        # Original + LoRA
        result = self.linear(x)
        lora_result = x @ self.lora_A.T @ self.lora_B.T
        return result + lora_result 


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    Now with LoRA on FFN layers.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 attention_dropout=0.1,
                 drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.pre_norm = LayerNorm(d_model)
        # self.self_attn = AttentionwLora(dim=d_model,
        #                            num_heads=nhead,
        #                            attention_dropout=attention_dropout,
        #                            projection_dropout=dropout)
        self.self_attn = Attention(dim=d_model,
                                   num_heads=nhead,
                                   attention_dropout=attention_dropout,
                                   projection_dropout=dropout)

        # FFN layers with LoRA (rank=8)
        # self.linear1 = LinearwLora(d_model, dim_feedforward)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        # self.linear2 = LinearwLora(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)
        self.gelu_bias = nn.Parameter(torch.zeros(dim_feedforward))

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()
        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Attention block
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)


        linear1_output = self.linear1(src) 
        
        # Step 2: Dropout
        dropout1_output = self.dropout1(linear1_output)
        

        biased_output = dropout1_output + self.gelu_bias 
        

        gelu_output = self.activation(biased_output)
        

        linear2_output = self.linear2(gelu_output)
        

        src2 = self.dropout2(linear2_output)
        src = src + self.drop_path(src2)
        
        return src

class MaskedTransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 attention_dropout = 0.1,
                 drop_path_rate = 0.1):
        super(MaskedTransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = MaskedAttentionWithLoRA(dim = d_model,
                                         num_heads = nhead,
                                         attention_dropout = attention_dropout,
                                         projection_dropout = dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, mask = None, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src), mask))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class TransformerClassifier(Module):

    def __init__(self,
                 seq_pool = True,
                 embedding_dim = 768,
                 num_layers = 12,
                 num_heads = 12,
                 mlp_ratio = 4.0,
                 num_classes = 1000,
                 dropout = 0.1,
                 attention_dropout = 0.1,
                 stochastic_depth = 0.1,
                 positional_embedding = 'learnable',
                 sequence_length = None):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad = True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim), requires_grad = True)
                init.trunc_normal_(self.positional_emb, std = 0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad = False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p = dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model = embedding_dim,
                                    nhead = num_heads,
                                    dim_feedforward = dim_feedforward,
                                    dropout = dropout,
                                    attention_dropout = attention_dropout,
                                    drop_path_rate = dpr[i]) for i in range(num_layers)
        ])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode = 'constant', value = 0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim = 1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim = 1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std = .02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000**(2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class MaskedTransformerClassifier(Module):

    def __init__(self,
                 seq_pool = True,
                 embedding_dim = 768,
                 num_layers = 12,
                 num_heads = 12,
                 mlp_ratio = 4.0,
                 num_classes = 1000,
                 dropout = 0.1,
                 attention_dropout = 0.1,
                 stochastic_depth = 0.1,
                 positional_embedding = 'sine',
                 seq_len = None,
                 *args,
                 **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.seq_pool = seq_pool
        self.num_tokens = 0

        assert seq_len is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            seq_len += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad = True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                seq_len += 1  # padding idx
                self.positional_emb = Parameter(torch.zeros(1, seq_len, embedding_dim), requires_grad = True)
                init.trunc_normal_(self.positional_emb, std = 0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(seq_len, embedding_dim, padding_idx = True),
                                                requires_grad = False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p = dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            MaskedTransformerEncoderLayer(d_model = embedding_dim,
                                          nhead = num_heads,
                                          dim_feedforward = dim_feedforward,
                                          dropout = dropout,
                                          attention_dropout = attention_dropout,
                                          drop_path_rate = dpr[i]) for i in range(num_layers)
        ])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x, mask = None):
        if self.positional_emb is None and x.size(1) < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode = 'constant', value = 0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim = 1)
            if mask is not None:
                mask = torch.cat([torch.ones(size = (mask.shape[0], 1), device = mask.device), mask.float()], dim = 1)
                mask = (mask > 0)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, mask = mask)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim = 1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std = .02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim, padding_idx = False):
        pe = torch.FloatTensor([[p / (10000**(2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        if padding_idx:
            return torch.cat([torch.zeros((1, 1, dim)), pe], dim = 1)
        return pe
