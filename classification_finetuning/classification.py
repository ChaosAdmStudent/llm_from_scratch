''' 
In this file, I import the open-weights OpenAI model we prepared earlier and then use it for classification fine-tuning. 
'''

import torch 
import torch.nn as nn
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
from pretraining.pretrained_openai import load_weights_into_gpt 
from pretraining.gpt_download import download_and_load_gpt2 
from attention.multihead_attention import ModelArgs
from pretraining.pretraining import GPTModel
from pretraining.utils import generate, text_to_ids, tk_ids_to_text 
import tiktoken 
from pathlib import Path
from classification_finetuning.data_prep import prepare_pd_dataset, split_train_test_val, create_data_loader
from classification_finetuning.classification_metrics import calc_accuracy_loader, calc_loss_loader

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

    BASE_CONFIG.update(model_configs["gpt2-small (124M)"]) 
    model = GPTModel(BASE_CONFIG, kv_args) 
    model.eval() 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    # Load weights into model 
    settings, params = download_and_load_gpt2(model_size="124M", models_dir='ch5/gpt2') 
    load_weights_into_gpt(model, params) 
    model = model.to(device) 

    # Generate some text to see if it works 
    texts = ["Every effort moves you"] 
    tokenizer = tiktoken.get_encoding('gpt2') 
    input_token_ids = torch.tensor([tokenizer.encode(text) for text in texts])
     
    model.toggle_kv_cache(True)
    out_tk_ids = generate(
        max_new_tokens=25, 
        model= model, 
        input_token_embeddings=input_token_ids, 
        context_length=BASE_CONFIG["context_length"], 
        device=device, 
        use_kv_cache=True
    ) 

    # Freeze all model parameters 
    for param in model.parameters(): 
        param.requires_grad_(False) 

    # Update output head for classification 
    out_classes = 2 
    classification_head = nn.Linear(BASE_CONFIG["token_emb_dim"], out_classes)  # Has requires_grad = True by default
    model.out_head = classification_head 

    # Unfreeze last transformer block and Final Layer norm parameters for better predictive performance 
    for param in model.trf_blocks[-1].parameters(): 
        param.requires_grad_(True) 
    for param in model.final_norm.parameters(): 
        param.requires_grad_(True) 
    
    model.toggle_kv_cache(False)
    model = model.to(device) 

    # Create data loaders 
    extracted_path = "ch6/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv" 
    df = prepare_pd_dataset(data_file_path)    
    df_train, df_test, df_val = split_train_test_val(df)

    pad_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    batch_size = 8
    shuffle = True 
    drop_last = True 
    num_workers = 2 

    train_loader = create_data_loader(df_train, tokenizer, max_length=None, pad_token=pad_token, batch_size=batch_size, 
                                      shuffle=shuffle, drop_last=drop_last, num_workers=num_workers) 

    test_loader = create_data_loader(df_test, tokenizer, max_length=None, pad_token=pad_token, batch_size=batch_size, 
                                      shuffle=shuffle, drop_last=drop_last, num_workers=num_workers) 
    
    val_loader = create_data_loader(df_val, tokenizer, max_length=None, pad_token=pad_token, batch_size=batch_size, 
                                      shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


    # Find classification accuracy on loader before fine-tuning
    print(f'Train loader classification accuracy PRE fine-tuning: {calc_accuracy_loader(train_loader, model, device)* 100:.2f}%') 
    print(f'Val loader classification accuracy PRE fine-tuning: {calc_accuracy_loader(val_loader, model, device)* 100:.2f}%')
    print(f'Test loader classification accuracy PRE fine-tuning: {calc_accuracy_loader(test_loader, model, device)* 100:.2f}%')

