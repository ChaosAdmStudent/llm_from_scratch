'''
I analyze the effect of the scale factor on probability distribution. This concept is used in "Decoding Strategies" for Temperature Scaling.
'''

import torch 

logits = torch.tensor([4,10,15], dtype=torch.float32) 

print('Logits: ', logits) 
original_probas = torch.softmax(logits, dim=-1) 
print('Original probabilities: ', original_probas) 

less_1_scale_probas =  torch.softmax(logits/0.1, dim=-1) 
print('Probabilities after dividing logits by 0.1: ', less_1_scale_probas) 

greater_1_scale_probas = torch.softmax(logits/5, dim=-1) 
print('Probabilities after dividing logits by 5: ', greater_1_scale_probas)