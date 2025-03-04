'''
In this file, I code the gpt model which includes the input embeddings, multiple stacked transformer blocks, followed by the output head. 
A single transformer block includes 1 masked multi-head attention module I coded before. 
'''

import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import torch.nn as nn 
import tiktoken
from ch4.utils import GELU, generate_new_tokens
from ch3.multihead_attention import MultiHeadAttention_V2, ModelArgs

class GPTModel(nn.Module): 
    def __init__(self, cfg: dict, kv_args: ModelArgs, use_kv_cache=False): 
        super(GPTModel, self).__init__() 

        # Token and position embedding layers 
        self.token_emb_layer = nn.Embedding(cfg["vocab_size"], cfg["token_emb_dim"]) 
        self.pos_emb_layer = nn.Embedding(cfg["context_length"], cfg["token_emb_dim"])  

        # Dropout layer for generated input embeddings 
        self.drop_inp_emb = nn.Dropout(cfg["droprate"])

        # Transformer blocks 
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, kv_args, use_kv_cache) for _ in range(cfg["num_layers"])]
        )

        # Layer norm 
        self.final_norm = LayerNorm(cfg["token_emb_dim"])

        # Output prediction head 
        self.out_head = nn.Linear(cfg["token_emb_dim"], cfg["vocab_size"]) 
    
    def forward(self, x, start_pos: int = None):
        """
        x: Tokenized text. Will have shape (B, num_tokens) 
        output: (B, num_tokens, vocab_size) 
        """   
        assert len(x.shape) == 2, "Input must be of shape (B, num_tokens)"
        B, num_tokens = x.shape 

        token_embeddings = self.token_emb_layer(x) # (B, num_tokens, token_emb_dim) 
        pos_embeddings = self.pos_emb_layer(torch.arange(num_tokens, device=x.device)) 
        input_embeddings = token_embeddings + pos_embeddings 

        # Dropout on input embeddings 
        input_embeddings = self.drop_inp_emb(input_embeddings)  

        # Pass through transformer blocks 
        out = input_embeddings
        for trf_block in self.trf_blocks: 
            out = trf_block(out, start_pos)  

        # Pass through layer norm 
        out = self.final_norm(out) 

        # Pass through prediction head 
        out = self.out_head(out) 

        return out 

class TransformerBlock(nn.Module): 
    def __init__(self, cfg, kv_args: ModelArgs, use_kv_cache = False): 
        super(TransformerBlock, self).__init__()  
        self.layer_norm1 = LayerNorm(cfg["token_emb_dim"]) 
        self.kv_args = kv_args # Parameters required for KV Cache
        self.use_kv_cache = use_kv_cache
        self.att = MultiHeadAttention_V2(
            cfg["token_emb_dim"], 
            cfg["token_emb_dim"], 
            cfg["context_length"], 
            cfg["droprate"], 
            cfg["num_heads"], 
            cfg["qkv_bias"] 
            ) 
        self.dropout = nn.Dropout(cfg["droprate"]) 
        self.layer_norm2 = LayerNorm(cfg["token_emb_dim"]) 
        self.ff = FeedForward(cfg["token_emb_dim"]) 

    def forward(self, x, start_pos: int = None): 

        assert not self.training and start_pos is None, "Must provide start_pos during inference for using KV Cache!" 

        if self.training: 
            self.att.kv_cache_enabled = False  
        elif not self.training and self.use_kv_cache: 
            self.att.kv_cache_enabled = True
            print('KV Cache enabled!')  

        res = x # First res connection
        out = self.layer_norm1(x) # (B,N,token_emb) 
        out = self.att(out, self.kv_args, start_pos) # (B,N, token_emb) 
        out = self.dropout(out) # (B, N, token_emb) 
        out = out + res # Res connection # (B,N, token_emb) 

        res = out # Second res connection
        out = self.layer_norm2(out) # (B, N, token_emb) 
        out = self.ff(out) # (B, N, token_emb) 
        out = self.dropout(out) # (B,N,token_emb) 
        out = out + res # Res connection # (B,N,token_emb) 

        return out 
    
class LayerNorm(nn.Module): 
    def __init__(self, emb_dim: int, eps=1e-5): 
        super(LayerNorm, self).__init__() 
        self.scale = nn.Parameter(torch.ones(emb_dim)) 
        self.shift = nn.Parameter(torch.zeros(emb_dim)) 
        self.eps = eps 

    def forward(self, x: torch.Tensor): 
        # x has shape (B, N, emb_dim) 

        mean =  x.mean(-1, keepdim=True) # (B,N,1) 
        var = x.var(-1, keepdim=True,unbiased=False) # (B,N,1) Unbiased = False divides by constant "n" in variance formula instead of "n-1" (Bessel's Correction) 
        normalized_x = (x-mean)/torch.sqrt(var + self.eps) # (B,N,emb_dim) 

        return self.scale * normalized_x + self.shift  

class FeedForward(nn.Module):  # Feed Forward modules help a lot in model understanding and genearlization because it allows exploration of a richer representation since we are expanding dimensions
    def __init__(self, emb_dim): 
        super(FeedForward, self).__init__()
        self.ff1 = nn.Linear(emb_dim, 4*emb_dim) 
        self.act = GELU() 
        self.ff2 = nn.Linear(4*emb_dim, emb_dim)  

        self.layers = nn.Sequential(self.ff1, self.act, self.ff2)
     
    def forward(self, x): 
        return self.layers(x) 

if __name__ == '__main__': 
    GPT_CONFIG_124M = {
        "token_emb_dim": 768, 
        "droprate": 0.1, 
        "vocab_size": 50257, 
        "context_length": 1024, 
        "num_heads": 12, 
        "num_layers": 12, 
        "qkv_bias": False 
    }   

    torch.manual_seed(123) 
    torch.set_printoptions(sci_mode=False) # To avoid printing super tiny numbers like 1e-8 and replace with 0

    tokenizer = tiktoken.get_encoding('gpt2') # BPE made of 50257 vocab size 

    text1 = "Every effort moves you" 
    text2 = "Every day is a" 
    token_ids = [torch.tensor(tokenizer.encode(text)) for text in [text1, text2]]
    token_ids = torch.stack(token_ids, dim=0) 

    # Testing GPT Model
    gpt_model = GPTModel(GPT_CONFIG_124M) 
    gpt_model.to(token_ids.device)

    # out = gpt_model(token_ids) 
    # print('Input tokens shape: ', token_ids.shape) 
    # print('GPT Output shape: ',out.shape)  

    # # Ex 4.1 

    # feed_forward = FeedForward(768) 
    # multihead_att = MultiHeadAttention_V2(
    #     GPT_CONFIG_124M["token_emb_dim"], 
    #     GPT_CONFIG_124M["token_emb_dim"], 
    #     GPT_CONFIG_124M["context_length"], 
    #     GPT_CONFIG_124M["droprate"], 
    #     GPT_CONFIG_124M["num_heads"], 
    #     GPT_CONFIG_124M["qkv_bias"]
    #     )
    
    # print('Number of parameters in FeedForward Module: ', sum(p.numel() for p in feed_forward.parameters()))
    # print('Number of parameters in MultiHead Attention Module: ', sum(p.numel() for p in multihead_att.parameters()))
    

    # Using text generator 
    gpt_model.eval()
    new_token_ids = generate_new_tokens(gpt_model, 6, token_ids, GPT_CONFIG_124M["context_length"]) 
    print(f'Before generating 6 tokens, shape: {token_ids.shape}') 
    print(f'After generating 6 tokens, shape: {new_token_ids.shape}') 
    print('Initial token IDs: \n', token_ids) 
    print('After generating new tokens: \n', new_token_ids) 

    sentences = [tokenizer.decode(sentence.tolist()) for sentence in new_token_ids] 
    print(sentences) 