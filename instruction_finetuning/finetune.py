import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch 
import torch.nn as nn 
import tiktoken 
from model_architecture.gpt_model import GPTModel 
from attention.multihead_attention import ModelArgs
from pretraining.gpt_download import download_and_load_gpt2 
from pretraining.pretrained_openai import load_weights_into_gpt 
from pretraining.utils import generate
from instruction_finetuning.data_prep import format_input_alpaca

if __name__ == '__main__': 
    # Load base model 
    BASE_CONFIG = {
        "vocab_size": 50257, 
        "context_length": 1024, 
        "droprate": 0.0, 
        "qkv_bias": True
    }
    
    model_configs = {
        "gpt2-small (124M)": {"token_emb_dim": 768, "num_layers": 12, "num_heads": 12},
        "gpt2-medium (355M)": {"token_emb_dim": 1024, "num_layers": 24, "num_heads": 16},
        "gpt2-large (774M)": {"token_emb_dim": 1280, "num_layers": 36, "num_heads": 20},
        "gpt2-xl (1558M)": {"token_emb_dim": 1600, "num_layers": 48, "num_heads": 25},
    } 

    kv_args = ModelArgs()
    BASE_CONFIG.update(model_configs["gpt2-medium (355M)"]) 
    model = GPTModel(BASE_CONFIG, kv_args) 
    model.eval() 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # Load weights into model 
    settings, params = download_and_load_gpt2(model_size="355M", models_dir='pretraining/gpt2') 
    load_weights_into_gpt(model, params) 
    model = model.to(device) 

    # Model output before fine-tuning 
    entry = {'instruction': 'Convert the active sentence to passive', 
             'input': f'\'The chef cooks the meal every day.\''} 
    
    tokenizer = tiktoken.get_encoding('gpt2') 
    texts = format_input_alpaca(entry)
    input_token_ids = torch.tensor([tokenizer.encode(text) for text in texts])
     
    model.toggle_kv_cache(True)
    out_tk_ids = generate(
        max_new_tokens=35, 
        model= model, 
        input_token_embeddings=input_token_ids, 
        context_length=BASE_CONFIG["context_length"], 
        device=device, 
        use_kv_cache=True
    ) 

    print(out_tk_ids.shape) 