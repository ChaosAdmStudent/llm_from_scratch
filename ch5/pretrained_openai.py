'''
In this file, I load open weights from OpenAI and use them to directly carry out inference on my model 
'''
import torch 
import torch.nn as nn
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ch5.gpt_download import download_and_load_gpt2 
import numpy as np
from ch5.pretraining import GPT_CONFIG_124M, GPTModel
from ch5.utils import generate
import tiktoken
from ch3.multihead_attention import ModelArgs
from ch5.inference import run_benchmarks
import gc

def assign(left: torch.Tensor, right: torch.Tensor): 
    if left.shape != right.shape: 
        raise ValueError(f"Shape mismatch. Left shape: {left.shape}, Right shape: {right.shape}") 
    else: 
        return nn.Parameter(torch.tensor(right)) 
    
def load_weights_into_gpt(gpt, params):
    """
    gpt: My from-scratch GPT model architecture
    params: Pretrained gpt model open weights from OpenAI
    """

    gpt.pos_emb_layer.weight = assign(gpt.pos_emb_layer.weight, params['wpe'])
    gpt.token_emb_layer.weight = assign(gpt.token_emb_layer.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.proj_out.weight = assign(
            gpt.trf_blocks[b].att.proj_out.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.proj_out.bias = assign(
            gpt.trf_blocks[b].att.proj_out.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].layer_norm1.scale = assign(
            gpt.trf_blocks[b].layer_norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].layer_norm1.shift = assign(
            gpt.trf_blocks[b].layer_norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].layer_norm2.scale = assign(
            gpt.trf_blocks[b].layer_norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].layer_norm2.shift = assign(
            gpt.trf_blocks[b].layer_norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

if __name__ == '__main__': 

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    kv_args = ModelArgs()

    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG, kv_args)
    gpt.eval() 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Download and load weights
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="ch5/gpt2") 
    load_weights_into_gpt(gpt, params) 
    gpt.to(device) 

    # Check output generated text 
    torch.manual_seed(123) 
    texts = ["Every effort moves you"] 
    tokenizer = tiktoken.get_encoding('gpt2') 
    input_token_ids = torch.tensor([tokenizer.encode(text) for text in texts])

    # Low temp and low topK 
    # temperature = 0.1   
    # top_k = 10
    # max_new_tokens = 25

    # output_tokens = generate(max_new_tokens, gpt, input_token_ids, NEW_CONFIG["context_length"], device, temperature, top_k) 
    # print(f'Temperature: {temperature}, topK: {top_k}') 
    # print('--------------------------------------------')
    # print([tokenizer.decode(output_batch.tolist()) for output_batch in output_tokens]) 
    # print('--------------------------------------------')

    # # High temp and high topK 
    # temperature = 5
    # top_k = 100

    # output_tokens = generate(max_new_tokens, gpt, input_token_ids, NEW_CONFIG["context_length"], device, temperature, top_k) 
    # print(f'Temperature: {temperature}, topK: {top_k}') 
    # print('--------------------------------------------')
    # print([tokenizer.decode(output_batch.tolist()) for output_batch in output_tokens]) 
    # print('--------------------------------------------') 

    # temperature = 1.5
    # top_k = 50

    # output_tokens = generate(max_new_tokens, gpt, input_token_ids, NEW_CONFIG["context_length"], device, temperature, top_k) 
    # print(f'Temperature: {temperature}, topK: {top_k}') 
    # print('--------------------------------------------')
    # print([tokenizer.decode(output_batch.tolist()) for output_batch in output_tokens]) 
    # print('--------------------------------------------')   

    # Compare inference speeds with larger contexts 
    # del output_tokens, input_token_ids 
    # torch.cuda.empty_cache()
    # gc.collect() 

    # with open('ch2/the-verdict.txt', 'r') as book: 
    #     raw_text = book.read() 

    # texts = [raw_text[:5], raw_text[:15],raw_text[:100], raw_text[:600], raw_text[:1500], raw_text[:4000]]
    # # texts = [raw_text[:100], raw_text[:600], raw_text[:4000]]
    # max_new_tokens = 30 
    # out_folder = 'ch5/plots'
    # seq_lens, kv_times, no_kv_times = run_benchmarks(gpt, GPT_CONFIG_124M , texts, tokenizer, device, max_new_tokens, out_folder) 
    
    # print('Sequence lengths: ',seq_lens) 
    # print('Total KV times: ',kv_times)
    # print('Total No KV times: ',no_kv_times)

    # Check if KV Cache is able to reproduce outputs like non KV
    temperature = 0 
    top_k = None
    max_new_tokens = 25

    output_tokens = generate(max_new_tokens, gpt, input_token_ids, NEW_CONFIG["context_length"], device, temperature, top_k, use_kv_cache=False) 
    print(f'Without KV Cache') 
    print('--------------------------------------------')
    print([tokenizer.decode(output_batch.tolist()) for output_batch in output_tokens]) 
    print('--------------------------------------------')

    gpt.toggle_kv_cache(True) 
    output_tokens = generate(max_new_tokens, gpt, input_token_ids, NEW_CONFIG["context_length"], device, temperature, top_k, use_kv_cache=True) 
    print(f'With KV Cache') 
    print('--------------------------------------------')
    print([tokenizer.decode(output_batch.tolist()) for output_batch in output_tokens]) 
    print('--------------------------------------------')
    








