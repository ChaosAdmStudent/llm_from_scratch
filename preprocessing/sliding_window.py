'''
Creating input-target pairs for LLM using sliding window approach 
''' 

import tiktoken
import torch 
from torch.utils.data import DataLoader, Dataset

class GPTDatasetV1(Dataset): 

    def __init__(self, txt, tokenizer, max_length, stride): 
        self.input_ids = [] 
        self.target_ids = [] 

        self.tokenized = tokenizer.encode(txt) 
        self.max_length = max_length 
        self.stride = stride

        for i in range(0, len(self.tokenized) - self.max_length, self.stride): 
            x = self.tokenized[i:i+self.max_length] 
            y = self.tokenized[i+1: i+1+self.max_length] 
            self.input_ids.append(torch.tensor(x, dtype=torch.long)) 
            self.target_ids.append(torch.tensor(y, dtype=torch.long))

    def __len__(self): 
        return len(self.input_ids)  

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]  

def create_dataloader(txt, tokenizer, max_length=10, stride=1, batch_size=1, num_workers=0, drop_last=True, shuffle=True): 
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) 
    torch.manual_seed(121) 
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        drop_last=drop_last 
    ) 

if __name__ == '__main__': 
    with open('ch2/the-verdict.txt','r') as book: 
        raw_text = book.read() 

    bpe = tiktoken.get_encoding('gpt2')  
    max_length = 6
    stride = 4
    batch_size = 2

    dataloader = create_dataloader(raw_text, bpe, max_length, stride, batch_size, shuffle=False) 
    data_iter = iter(dataloader) 
    x,y = next(data_iter) 

    print(x) 
    print(y)
    

