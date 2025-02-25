'''
In this file, I write code for pretraining the LLM. 
'''

import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch 
import torch.nn as nn
import tiktoken 
from ch4.gpt_model import GPTModel
from ch5.utils import * 
from ch5.loss import cross_entropy_loss
from ch2.sliding_window import create_dataloader
from ch5.utils import  generate

def calc_loss_batch(input_batch, target_batch, model, device): 
    """
    Calculates loss of a single batch 
    """
    input_batch = input_batch.to(device) 
    target_batch = target_batch.to(device) 
    logits = model(input_batch) # (B, N, vocab_size) 

    # Pytorch CE function does softmax on input, takes out values at indices provided in target_batch; uses them in CE formula
    return nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten()) # This way input has shape (B*N, vocab_size) and target has shape (B*N,)

def calc_loss_loader(data_loader, model, device, num_batches = None): 
    """
    Calculates loss over a whole data loader. Used for evaluation purposes
    num_batches: Number of batches to use for average computation of loss. By default, averages over all samples in a batch 
    """
    total_loss = 0 
    if len(data_loader) == 0: 
        return torch.float("nan") 
    elif num_batches == None: 
        num_batches = len(data_loader)
    else: 
        num_batches = min(len(data_loader), num_batches) 
    
    for i, (input_batch, target_batch) in enumerate(data_loader): 
        if i < num_batches: 
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item() 
        else: 
            break 
    
    return total_loss/num_batches

def evaluate_model(train_loader, val_loader, model, device, num_batches=None): 
    model.eval() # Disables training specific layers like Dropout 
    with torch.no_grad(): # Avoids computational graph overhead
        train_loss = calc_loss_loader(train_loader, model, device, num_batches) 
        val_loss = calc_loss_loader(val_loader, model, device, num_batches) 
    
    model.train() # Switches back to training mode so that training loop is unaffected
    return train_loss, val_loss 

if __name__ == '__main__': 
    tokenizer = tiktoken.get_encoding('gpt2') 
    GPT_CONFIG_124M = {
        "token_emb_dim": 768, 
        "droprate": 0.1, 
        "vocab_size": 50257, 
        "context_length": 4, 
        "num_heads": 12, 
        "num_layers": 4, 
        "qkv_bias": False 
    }    

    torch.manual_seed(123) 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device: ', device) 

    model = GPTModel(GPT_CONFIG_124M) 
    model = model.to(device) 

    # Read raw text corpus 
    with open('ch2/the-verdict.txt', 'r') as book: 
        raw_text = book.read() 
    
    # Create training and testing split 
    train_ratio = 0.9 
    train_idx = int(0.9 * len(raw_text))
    train_text = raw_text[:train_idx] 
    val_text = raw_text[train_idx: ] 

    # Tokenize raw corpus 
    tokenizer = tiktoken.get_encoding('gpt2') 

    # Create dataloaders 
    train_loader = create_dataloader(
        train_text, 
        tokenizer,
        max_length=GPT_CONFIG_124M["context_length"], 
        stride=GPT_CONFIG_124M["context_length"], 
        batch_size=8, 
        drop_last=True, 
        shuffle=True
    )

    val_loader = create_dataloader(
        val_text, 
        tokenizer,
        max_length=GPT_CONFIG_124M["context_length"], 
        stride=GPT_CONFIG_124M["context_length"], 
        batch_size=2, 
        drop_last=True, 
        shuffle=True
    ) 

    # Optimizer 
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1) # AdamW has better weight decay implementation than Adam. Heavily used in LLMs 

    # Training loop 
    model.train() 
    for epoch in range(10): 
        for i, (x_train, y_train) in enumerate(train_loader): 
            if i == 0: # For visualization purposes. 
                print('Sample Inputs: ', tokenizer.decode(x_train[0].tolist())) 
                print('Sample Outputs: ', tokenizer.decode(y_train[0].tolist())) 
            
            # Calculate CE loss 
            loss = calc_loss_batch(x_train, y_train, model, device) 
            perplexity = torch.exp(loss).item() 
            
            loss.backward() # Backpropagate to calculate gradients 
            optimizer.step() # Perform gradient descent with commputed param gradients 
            optimizer.zero_grad() # Clear accumulated gradients  

            if i % 3: 
                print('Saving model and optimizer...') 
                
                # Save model 
                torch.save({
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict()
                }, 'ch5/model_checkpoint.pth')  

                # Load model 
                model = GPTModel(GPT_CONFIG_124M) 
                model = model.to(device) 
                optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1) 

                checkpoint = torch.load('ch5/model_checkpoint.pth', map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print('Model loaded again!')

            if i % 10 == 0: 
                train_loss, val_loss = evaluate_model(train_loader, val_loader, model, device, 10)
                print(f'\t epoch {epoch}/10 batch {i}/{len(train_loader)} Train loss: {train_loss}, Eval loss: {val_loss}')   


    # Decoding strategies 
    model.eval()  
    max_new_tokens = 25 
    input_token_embeddings = torch.tensor([tokenizer.encode(text) for text in ["I love your", "Do you know"]]) 
    generated_tokens = generate(max_new_tokens, model, input_token_embeddings, GPT_CONFIG_124M["context_length"], device, temperature=5, top_k=5)
    
    print(f"With temperature 5: ") 
    generated_text = [tokenizer.decode(token_ids.tolist()) for token_ids in generated_tokens] 
    print(generated_tokens) 
    print('---------------------------') 

    print(f"With temperature 0.1: ") 
    generated_tokens = generate(max_new_tokens, model, input_token_embeddings, GPT_CONFIG_124M["context_length"], device, temperature=0.1, top_k=5)
    generated_text = [tokenizer.decode(token_ids.tolist()) for token_ids in generated_tokens] 
    print(generated_tokens) 
    print('---------------------------')
