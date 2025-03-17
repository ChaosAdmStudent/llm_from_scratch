import torch 
import torch.nn as nn
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from torch.utils.data import DataLoader
from ch4.gpt_model import GPTModel

def calc_accuracy_loader(data_loader: DataLoader, model: GPTModel, device, num_batches=None): 
    total_examples, correct_preds = 0,0
    
    if num_batches == None: 
        num_batches = len(data_loader) 

    else: 
        num_batches = min(num_batches, len(data_loader)) 
    
    for i, (x_train, y_train) in enumerate(data_loader): 
        if  i < num_batches: 
            x_train = x_train.to(device) 
            y_train = y_train.to(device) 

            with torch.no_grad(): 
                output_logits = model(x_train) # (B,N,out_classes)  
            useful_logits = output_logits[:,-1,:] # Last token's logits will have info taking all tokens into account
            pred_labels = torch.argmax(useful_logits, dim=-1) # (B,) 

            total_examples += output_logits.shape[0] * output_logits.shape[1] 
            correct_preds += (pred_labels == y_train).sum().item() 

        else: 
            break 
    
    return correct_preds/total_examples

def calc_loss_batch(input_batch: torch.Tensor , target_batch: torch.Tensor, model, device): 
    """
    input_batch: (B, N) 
    target_batch: (B,) 
    """ 

    input_batch = input_batch.to(device) 
    target_batch = target_batch.to(device) 
    useful_logits = model(input_batch)[:, -1, :] # (B,2) 
    loss = nn.functional.cross_entropy(useful_logits, target_batch)

    return loss 
    

def calc_loss_loader(data_loader: DataLoader, model: GPTModel, device, num_batches=None): 
    total_loss = 0
    
    if num_batches == None: 
        num_batches = len(data_loader) 

    else: 
        num_batches = min(num_batches, len(data_loader)) 
    
    for i, (x_train, y_train) in enumerate(data_loader): 
        if  i < num_batches: 
            with torch.no_grad(): 
                total_loss += calc_loss_batch(x_train, y_train, model, device).item()   

        else: 
            break 
    
    return total_loss/num_batches