'''
I analyze the effect of the scale factor on probability distribution. This concept is used in "Decoding Strategies" for Temperature Scaling.
'''

import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  
from ch5.utils import print_sampled_tokens

# logits = torch.tensor([4,10,15], dtype=torch.float32) 

# print('Logits: ', logits) 
# original_probas = torch.softmax(logits, dim=-1) 
# print('Original probabilities: ', original_probas) 

# less_1_scale_probas =  torch.softmax(logits/0.1, dim=-1) 
# print('Probabilities after dividing logits by 0.1: ', less_1_scale_probas) 

# greater_1_scale_probas = torch.softmax(logits/5, dim=-1) 
# print('Probabilities after dividing logits by 5: ', greater_1_scale_probas)  

# Testing temp scaling 

vocab = {
    "closer": 0, 
    "every": 1, 
    "effort": 2, 
    "forward": 3, 
    "inches": 4, 
    "moves": 5, 
    "pizza": 6, 
    "toward": 7, 
    "you": 8
}
inverse_vocab = {v:k for k,v in vocab.items()}  
next_token_logits = torch.tensor( # Assume these are the next token logits 
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

# Temperature scaling 0<T<1 
temp_scale = 0.1 
T_logits = next_token_logits/temp_scale 
probas = torch.softmax(T_logits, dim=-1).unsqueeze(0) 
print('Temp scale: ', temp_scale) 
print_sampled_tokens(probas, inverse_vocab) 

# Temp scaling T>1 
temp_scale = 5 
T_logits = next_token_logits/temp_scale
probas = torch.softmax(T_logits, dim=-1).unsqueeze(0)  
print('Temp scale: ', temp_scale) 
print_sampled_tokens(probas, inverse_vocab) 