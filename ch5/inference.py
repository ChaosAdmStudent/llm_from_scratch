'''
This file has code to run inference on my trained model. To use openAI weights to do inference, use pretrained_openai.py. 
'''

import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import tiktoken 
from ch5.pretraining import GPT_CONFIG_124M 
from ch4.gpt_model import GPTModel
from ch3.multihead_attention import ModelArgs 
import torch 

# Initiate model 
use_kv_cache = True 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
kv_args = ModelArgs()
model = GPTModel(GPT_CONFIG_124M, kv_args, use_kv_cache=use_kv_cache) 
model.to(device) 

# Load model weights
checkpoint = torch.load('ch5/model_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print('Model loaded!') 

# Prepare inference prompts 
tokenizer = tiktoken.get_encoding('gpt2') # BPE 
max_new_tokens = 50 
text_prompts = [
    "Where do you", 
    "She is going",
    "My king is" 
] 

input_tokens = torch.tensor([tokenizer.encode(prompt) for prompt in text_prompts])  
print(input_tokens.shape) 