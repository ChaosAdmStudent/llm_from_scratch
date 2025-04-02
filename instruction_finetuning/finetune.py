#  TODO: Change generate function to stop generating if we encounter endoftext

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
from instruction_finetuning.data_prep import format_input_alpaca, download_and_load_file, split_train_val_test, create_dataloader_instruction
from instruction_finetuning.data_prep import collate_fn_dynamic_padding
from pretraining.pretraining import calc_loss_batch, calc_loss_loader, evaluate_model
from pathlib import Path
from instruction_finetuning.utils import generate_out_text_response, store_model_responses, store_openai_responses
from classification_finetuning.utils import plot_values

def train_instruction_finetune(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, start_context: str, context_length, tokenizer, num_batches=None):  
    print('###################################')
    print('Started training: ') 
    print('Monitoring given instruction: ') 
    print(start_context)
    print('###################################')  

    start_tokens = torch.tensor([tokenizer.encode(start_context)], device=device)  
    train_losses, val_losses, track_tokens_seen = [],[],[] 
    total_tokens_seen, global_step = 0, -1 

    train_loss, val_loss = evaluate_model(train_loader, val_loader, model, device, num_batches) 
    print('Train loss: ', train_loss) 
    print('Val loss: ', val_loss) 
    
    for epoch in range(num_epochs): 
        for i, (x_train, y_train) in enumerate(train_loader): 
            model.train() 
            x_train = x_train.to(device) 
            y_train = y_train.to(device) 
            loss = calc_loss_batch(x_train, y_train, model, device) 
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad() 

            total_tokens_seen += x_train.numel() 
            global_step += 1 

            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(train_loader, val_loader, model, device, num_batches) 
                train_losses.append(train_loss) 
                val_losses.append(val_loss) 
                track_tokens_seen.append(total_tokens_seen) 
                print(f'\t Epoch {epoch} step {global_step}, train_loss: {train_loss}, val_loss: {val_loss}')

        # Check how start_context is being replied to after each epoch 
        output_text = generate_out_text_response(model, input_text, start_tokens, context_length, tokenizer, device)
        print(f'#########Epoch {epoch}##############') 
        print(f'{output_text}')
        print(f'#########Epoch {epoch}##############')

    torch.save(model.state_dict(), 'instruction_finetuning/finetune_model_chk.pth')
    print('Model saved!')
    
    return train_losses, val_losses, track_tokens_seen

if __name__ == '__main__': 
    torch.manual_seed(123) 

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
    file_path = Path('instruction_finetuning/instruction-data.json') 
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)
    train_data, val_data, test_data = split_train_val_test(data) 
    tokenizer = tiktoken.get_encoding('gpt2') 
    input_text = format_input_alpaca(val_data[0])
    print('Input instruction:\n\t',input_text) 
    print('-----------------------------')
    input_token_ids = torch.tensor([tokenizer.encode(input_text)])  
    
    print('Before finetuning: ')
    print(generate_out_text_response(model, input_text, input_token_ids, BASE_CONFIG['context_length'], tokenizer, device)) 

    # Finetune training 
    batch_size = 8

    train_loader = create_dataloader_instruction(train_data, tokenizer, format_input_alpaca, collate_fn_dynamic_padding, 
                                                 batch_size=batch_size, device=device) 
    
    val_loader = create_dataloader_instruction(val_data, tokenizer, format_input_alpaca, collate_fn_dynamic_padding, 
                                                 batch_size=batch_size, device=device)  

    test_loader = create_dataloader_instruction(test_data, tokenizer, format_input_alpaca, collate_fn_dynamic_padding, 
                                                 batch_size=batch_size, device=device) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1) 
    num_epochs = 4
    eval_freq = 5
    num_batches = 5
    
    train_losses, val_losses, track_tokens_seen = train_instruction_finetune(model, train_loader, val_loader, 
                                                                             optimizer, device, num_epochs, eval_freq, input_text, 
                                                                             BASE_CONFIG['context_length'], tokenizer, num_batches)

    # Generate loss plot 
    epochs_seen_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_values(epochs_seen_tensor, torch.tensor(track_tokens_seen), train_losses, val_losses, label='loss', plot_dir='instruction_finetuning/plots')   

    # Model output after fine-tuning 
    print('After finetuning: ')
    print(generate_out_text_response(model, input_text, input_token_ids, BASE_CONFIG['context_length'], tokenizer, device)) 
    print('------------------------------------------------------------------')

    # Generate test data model outputs using fine-tuned model
    file_path = 'instruction_finetuning/finetune-responses.json' 
    print('Generating test data responses')
    store_model_responses(file_path, model, test_data, tokenizer, BASE_CONFIG['context_length'], device)  
    print(f'Test data responses created and stored in {file_path}')  

    # Generate test data model outputs using ChatGPT 
    store_openai_responses(test_data) 