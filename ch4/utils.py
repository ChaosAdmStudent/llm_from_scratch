import torch 
import torch.nn as nn 

class GELU(nn.Module): 
    def __init__(self): 
        super(GELU, self).__init__() 

    def forward(self, x): 
        return (0.5 * x * (1+ torch.tanh(
            torch.sqrt((2/torch.pi) * (x + 0.044715 * x**3))
        )
        )) 