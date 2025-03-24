import os 
import urllib.request
import json 
from pathlib import Path 
from typing import List
import random 
import torch 
from torch.utils.data import DataLoader, Dataset
from functools import partial
import tiktoken

def download_and_load_file(file_path, url): 

    if not os.path.exists(file_path): 
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data) 
    
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

def format_input_alpaca(entry: dict): 
    instruction_text = (
        "Below is an instruction that describes a task."
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction: {entry['instruction']}" 
    )

    input_text = f"\n\n### Input: {entry['input']}" if entry['input'] else "" 

    return instruction_text + input_text

def split_train_val_test(data: List, train_frac=0.7, val_frac=0.1): 
    random.shuffle(data) 
    train_end = int(train_frac * len(data)) 
    val_end = train_end + int(val_frac * len(data)) 

    train_data = data[:train_end] 
    val_data = data[train_end: val_end] 
    test_data = data[val_end: ] 

    return train_data, val_data, test_data

def create_dataloader_instruction(data, tokenizer, format_input_fn, collate_fn, batch_size=4, shuffle=True, drop_last=True, num_workers=0,
                                  pad_token=50256, device='cpu', ignore_index=-100, allowed_max_length=None, mask_instruction=False): 
    
    instr_dataset = InstructionDataset(data, tokenizer, format_input_fn)
    return DataLoader(
        instr_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        drop_last=drop_last, 
        collate_fn=partial(collate_fn,  # Partial creates a new function with the extra parameters baked into the new function which calls the provided collate_fn with them
                           pad_token=pad_token, device=device, ignore_index=ignore_index,
                           allowed_max_length=allowed_max_length, mask_instruction=mask_instruction)
    )


def collate_fn_dynamic_padding(batch, pad_token=50256, device='cpu', ignore_index=-100, allowed_max_length=None, mask_instruction=False):  
    """
    Input: 
        - Batch of variable sized tokenized formatted prompts returned by InstructionDataset 
        - ignore_index: Value to use in place of padded tokens 
        - allowed_max_length: Optional max length. If dataset size is more than model's context_length, this can be used 
    Output: Two tensors: 
        - Tensor 1: Batch of tokenized formatted prompts padded to length of largest tokenized prompt in this batch
        - Tensor 2: Tensor 1 shifted back by 1, the last token is replaced with padded token. This is our output that we want LLM to learn
    """
    
    # We increase max_length by 1, so that we can have one extra pad token.
    # This will make creating output tensor super easy
    max_batch_length = max([len(item[0]+item[1])+1 for item in batch])   
    inputs_lst = [] 
    outputs_lst = [] 

    for item in batch: 
        instruction, response = item 
        new_item = instruction + response 
        padded = new_item + [pad_token] * (max_batch_length - len(new_item)) 
        inputs = torch.tensor(padded[:-1]) # Removing the extra added padding for input 
        outputs = torch.tensor(padded[1:]) # Just shift back input by 1 and have pad token for the last one (the one we want model to generate at the end).

        # Replace padded tokens with ignore_index. Leave first padded token in each output_tensor because we want model to learn to generate 
        # end of sequence token as the last token 
        # We will code it in the training loop to make it ignore tensor values equal to ignore_index so that training is not influenced by padded tokens 
        mask = outputs == pad_token 
        indices = torch.nonzero(mask).squeeze() 
        if indices.numel() > 1:  # If we just have 1 padded token in output, that is needed for end-of-text prediction
            outputs[indices[1:]] = ignore_index 
        
        # Mask instruction tokens optionally 
        if mask_instruction: 
            outputs[:len(instruction)] = ignore_index

        if allowed_max_length is not None: 
            inputs = inputs[:allowed_max_length] 
            outputs = outputs[:allowed_max_length]

        inputs_lst.append(inputs) 
        outputs_lst.append(outputs) 

    inputs_tensor = torch.stack(inputs_lst).to(device) 
    outputs_tensor = torch.stack(outputs_lst).to(device)

    return inputs_tensor, outputs_tensor 

class InstructionDataset(Dataset): 
    def __init__(self, data: List, tokenizer, format_input_fn): 
        self.data = data  
        self.encoded_formatted_inputs = [tokenizer.encode(format_input_fn(entry)) for entry in data] 
        self.encoded_formatted_outputs = [tokenizer.encode(entry['output']) for entry in data]

    def __len__(self): 
        return len(self.encoded_formatted_inputs)

    def __getitem__(self, index):
        return self.encoded_formatted_inputs[index], self.encoded_formatted_outputs[index] 


if __name__ == '__main__': 
    file_path = Path('instruction_finetuning/instruction-data.json') 
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)

    list1 = ([5, 1, 3],  [4, 6])
    list2 = ([1, 3], [5])
    list3 = ([4], [2,3]) 

    batch = (
        list1, 
        list2, 
        list3
    )

    print(f'Batch: \n{batch}') 

    print('------------------------------')
    
    # Checking if masking padding tokens in target tensor works as intended
    print('Not masking instruction tokens: ')
    in_tensor, out_tensor = collate_fn_dynamic_padding(batch) 
    print(f'Input padded tensor: \n\t {in_tensor}') 
    print(f'Output padded tensor: \n\t {out_tensor}') 
    # Works!
    
    print('------------------------------')

    # Checking if masking instruction tokens work
    print('Masking instruction tokens: ')
    in_tensor, out_tensor = collate_fn_dynamic_padding(batch, mask_instruction=True) 
    print(f'Input padded tensor: \n\t {in_tensor}') 
    print(f'Output padded tensor: \n\t {out_tensor}')
    # Works!  

    print('------------------------------')

    # Checking if data loaders work and show variable size number of tokens for a given batch 
    train_data, val_data, test_data = split_train_val_test(data) 
    tokenizer = tiktoken.get_encoding('gpt2') 
    num_workers = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    train_loader = create_dataloader_instruction(train_data, tokenizer, format_input_alpaca, collate_fn_dynamic_padding, 
                                                 num_workers=num_workers, device=device) 

    print('Number of training batches: ', len(train_loader))
    for input, output in train_loader: 
        print(input.shape) 
    # Works: variable sized batches obtained
    
    