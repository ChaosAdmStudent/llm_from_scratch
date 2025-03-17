'''
I used this code to debug my KV implementation. The outputs were not matching between KV and non KV mode. 
After extensive debugging steps (can be seen below), I found the problem was with position embedding. I was adding the PE of 0th position to the new token input in KV mode everytime which messed up the input embedding and gave different results. 
'''

import torch
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretraining.gpt_download import download_and_load_gpt2 
import numpy as np
from pretraining.pretraining import GPT_CONFIG_124M, GPTModel
import tiktoken
from attention.multihead_attention import ModelArgs
from pretraining.pretrained_openai import load_weights_into_gpt 

def debug_generate_and_compare(
    model, input_tokens, block_index, check_step, max_steps=10, deterministic=True
):
    """
    Generate text with and without KV cache and compare Q, K, V at a given step.

    Args:
        model: GPT-like model
        input_tokens: Initial token sequence (torch tensor of shape [1, seq_len])
        block_index: Transformer block to debug (index)
        check_step: Step at which to compare Q, K, V
        max_steps: Number of tokens to generate
        deterministic: If True, uses argmax decoding (greedy); otherwise, allows stochasticity.

    Returns:
        None (prints out the differences)
    """

    def generate_tokens(model, input_tokens, use_kv_cache):
        """
        Runs a text generation loop and collects Q, K, V at 'check_step'.
        """
        generated_tokens = input_tokens.clone()
        process_tokens = generated_tokens 
        qkv_debug_data = None
        input_qkv_debug_data = None

        model.toggle_kv_cache(use_kv_cache)
        model.eval()
        with torch.no_grad():
            for step in range(max_steps):
                if step == 0 and use_kv_cache: 
                    start_pos = 0 
                else: 
                    start_pos = generated_tokens.shape[1] - 1 if use_kv_cache else None

                # Run the forward pass (capture Q, K, V at the desired block)
                logits, qkv_data, input_qkv, input_embeddings, token_embeddings = model(process_tokens, start_pos=start_pos, debug=True, debug_block_idx=block_index)

                if step == check_step:
                    qkv_debug_data = qkv_data  # Store Q, K, V tensors at check_step
                    input_qkv_debug_data = input_qkv
                    input_embeddings_debug_data = input_embeddings
                    token_embeddings_debug_data = token_embeddings

                # Get the next token (deterministic or sampled)
                if deterministic:
                    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # Greedy decoding
                else:
                    probs = torch.softmax(logits[:, -1] / 1.0, dim=-1)  # Apply temperature=1.0
                    next_token = torch.multinomial(probs, num_samples=1)  # Sample token

                # Append the generated token
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

                if use_kv_cache: 
                    process_tokens = next_token 
                else: 
                    process_tokens = generated_tokens 

                if step == check_step: 
                    print(process_tokens) 

        return generated_tokens, qkv_debug_data, input_qkv_debug_data, input_embeddings_debug_data, token_embeddings_debug_data

    # Run generation WITHOUT KV cache
    print("Running generation 1 without KV cache...")
    tokens_no_kv, qkv_no_kv, input_qkv_no_kv, in_emb_no_kv, in_tokemb_no_kv = generate_tokens(model, input_tokens, use_kv_cache=False)

    # Run generation WITH KV cache
    print("Running generation with KV cache...")
    tokens_with_kv, qkv_with_kv, input_qkv_with_kv, in_emb_kv, in_tokemb_kv = generate_tokens(model, input_tokens, use_kv_cache=True)

    # Compare the input token embeddings. 
    print('Input token embedding with KV: ', in_tokemb_kv[0, :, 0:5])  
    print('Input token embedding without KV: ', in_tokemb_no_kv[0,:, 0:5])

    # Compare the total input embeddings. Non-KV's last token's embedding should be same as that of KV's 
    print('Input total embedding with KV: ', in_emb_kv[0, :, 0:5])  
    print('Input total embedding without KV: ', in_emb_no_kv[0,:, 0:5])

    # Compare the generated outputs
    print("\nGenerated tokens without KV cache:", tokens_no_kv)
    print("Generated tokens with KV cache:", tokens_with_kv)

    # Compare Q, K, V at the debug step
    print("\nComparing Q, K, V tensors at block", block_index, "after step", check_step)

    if qkv_no_kv is not None and qkv_with_kv is not None:
        Q_no, K_no, V_no = qkv_no_kv
        Q_kv, K_kv, V_kv = qkv_with_kv

        shapes = Q_no.shape 
        Q_no = Q_no.transpose(1,2).view(shapes[0], shapes[2], shapes[1]*shapes[3])  
        K_no = K_no.transpose(1,2).view(shapes[0], shapes[2], shapes[1]*shapes[3])  
        V_no = V_no.transpose(1,2).view(shapes[0], shapes[2], shapes[1]*shapes[3])  
        K_kv = K_kv.transpose(1,2).view(shapes[0], shapes[2], shapes[1]*shapes[3]) 
        V_kv = V_kv.transpose(1,2).view(shapes[0], shapes[2], shapes[1]*shapes[3]) 
        Q_kv = Q_kv.transpose(1,2).view(Q_kv.shape[0], Q_kv.shape[2], Q_kv.shape[1]*Q_kv.shape[3]) 

        Q_no_input, K_no_input, V_no_input = input_qkv_no_kv 
        Q_kv_input, K_kv_input, V_kv_input = input_qkv_with_kv 

        print('Final QKV shapes no KV: ',Q_no.shape, K_no.shape, V_no.shape) 
        print('Final QKV shapes with KV: ',Q_kv.shape, K_kv.shape, V_kv.shape)  

        print('Initial QKV shapes no KV: ', Q_no_input.shape, K_no_input.shape, V_no_input.shape) 
        print('Initial QKV shapes with KV: ', Q_kv_input.shape, K_kv_input.shape, V_kv_input.shape)

        print('Input Q with KV: ', Q_kv_input[0, 0, 0:5])  
        print('Input Q without KV: ', Q_no_input[0,:, 0:5])  

        print('Input K with KV: ', K_kv_input[0, 0, 0:5])  
        print('Input K without KV: ', K_no_input[0,:, 0:5]) 

        print('Final K with KV: ', K_kv[0, :, 0:5])  
        print('Final K without KV: ', K_no[0,:, 0:5]) 

        print("Max difference in K:", (K_no - K_kv).abs().max().item())
        print("Max difference in V:", (V_no - V_kv).abs().max().item())
        print("Max difference in Q:", (Q_no - Q_kv).abs().max().item())

        assert torch.allclose(K_no, K_kv, atol=1e-5), "Mismatch in K values!"
        assert torch.allclose(V_no, V_kv, atol=1e-5), "Mismatch in V values!"
        assert torch.allclose(Q_no, Q_kv, atol=1e-5), "Mismatch in Q values!"
        print("✅ Q, K, V match between cached and non-cached runs!")

    else:
        print("⚠️ Warning: Q, K, V data was not collected properly.")

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
    input_token_ids = torch.tensor([tokenizer.encode(text) for text in texts]).to(device) 

    debug_generate_and_compare(
        gpt, 
        input_token_ids, 
        block_index=0,   # Debug transformer block 2
        check_step=1,    # Compare Q, K, V at step 5
        max_steps=5,    # Generate up to 10 tokens
        deterministic=True  # Greedy decoding for reproducibility
    )