from typing import Optional, Tuple
import torch
from torch import nn

class SiglipConfig:
    def init(self,
             hidden_size = 768,
             intermediate_size = 3072,
             num_hidden_layers =12,
             num_attention_heads = 12,
             num_channels = 3,
             image_size = 224,
             patch_size = 16,
             layer_norm_eps = 1e-6,
             attention_dropout = 0.0,
             num_image_tokens: int = None,
             **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens
 
'''
the classes here are done in vision transformer too that ive done in (ViT) .py and .ipynb file but writing them again here as we need ViT for siglip
'''

class SiglipVisionEmbedding(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_embed = nn.Conv2d(config.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, padding="valid")
        self.num_patches = (self.image_size//self.patch_size)**2
        self.num_position = self.num_patches
        self.position_embed = nn.Embedding(self.num_position, self.embed_dim)
        self.register_buffer("position_ids",
                            torch.arange(self.num_position).expand((1, -1)),
                            persistent= False)
        
    def forward(self, pixel_value: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_value.shape
        # how explained in ViT .py file
        patch_embed = self.patch_embed(pixel_value)
        embed = patch_embed.flatten(2)
        embed = embed.transpose(1,2)
        embed = embed + self.position_embed(self.position_ids)
        return embed

class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**0.5
        self.dropout = config.attention_dropout
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_state.size()
        query_states = self.q_proj(hidden_state)
        key_states = self.k_proj(hidden_state)
        value_states = self.v_proj(hidden_state)
        #basically going from ([batch_size, num_patches, embed_dim]) to ([batch_size, num_heads, num_patches, head_dim])
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        attn_weights = (torch.matmul(query_states, key_states.transpose(2,3))* self.scale)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training) # not really required, it isnt in paper but useful during training for overfitting
        attn_output = torch.matmul(attn_weights, value_states)
        
        # we go from ([batch_size, num_heads, num_patches, head_dim]) to ([batch_size, num_patches, num_heads, head_dim])
        attn_output = attn_output.transpose(1,2).contiguous()
        # then from ([batch_size, num_patches, num_heads, head_dim]) to ([batch_size, num_patches, embed_dim])
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights
        
        'so we started with (4, 1024) and ended with (4, 1024) dim'
    
class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.l1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ln2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.l1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate= "tanh")
        hidden_states = self.ln2(hidden_states)
        return hidden_states
    
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.ln1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.ln2 = nn.LayerNorm(self.embed_dim, eps= config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        res = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = res+ hidden_states
        res = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states

class SiglipVisionEncoder(nn.Module):
    def __init__(self, config: SiglipConfig):
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        
        return hidden_states

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embed = SiglipVisionEmbedding(config)
        self.encoder  = SiglipVisionEncoder(config) # this is literally the whole vision transformer i coded before lol (did it again here fully)
        self.post_ln = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
        
    def forward(self, pixel_value: torch.Tensor) -> torch.Tensor:
        # pixel_value dim = ([batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim])
        hidden_state = self.embed(pixel_value)
        last_hidden_state = self.encoder(inputs_embeds = hidden_state)
        last_hidden_state = self.post_ln(last_hidden_state)
        return last_hidden_state
        
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_value) -> Tuple:
        # same like the above one -([batch_size, channels, height, width] -> [batch_size, num_patches, embed_dim])
        return self.vision_model(pixel_value = pixel_value)