'''
In this file, I test how the gradients look like if we use skip connections in a feed forward network 
''' 
import torch 
import torch.nn as nn 

class DummySkipConnectionNetwork(nn.Module): 
    def __init__(self, layer_sizes, use_shortcut=True): 
        super(DummySkipConnectionNetwork, self).__init__() 
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(layer_sizes[0], layer_sizes[1]), nn.ReLU() 
            ),

            nn.Sequential(
                nn.Linear(layer_sizes[1], layer_sizes[2]), nn.ReLU() 
            ), 
            
            nn.Sequential(
                nn.Linear(layer_sizes[2], layer_sizes[3]), nn.ReLU() 
            )] 
        ) 
    
    def forward(self, x): 
        for layer in self.layers: 
            out = layer(x) 
            if self.use_shortcut and x.shape == out.shape: 
                x = out + x # Skip connection 
            else: 
                x = out
        return out 
    
def print_gradients(model, inputs): 
    out = model(inputs) 
    target = torch.tensor([[0.0]]) 

    loss = nn.functional.mse_loss(out, target) 

    loss.backward() 

    for name, param in model.named_parameters(): 
        if 'weight' in name: 
            print(f'{name} has gradient {param.grad.abs().mean().item()}') 

if __name__ == '__main__': 
    layer_sizes = [3,3,3,1] 
    batch = torch.rand(1,3) 

    torch.manual_seed(123)

    skip_network = DummySkipConnectionNetwork(layer_sizes, use_shortcut=True) 
    noskip_network = DummySkipConnectionNetwork(layer_sizes, use_shortcut=False) 
    
    out_skip = skip_network(batch) 
    out_noskip = noskip_network(batch) 

    print('Output shape with skip net: ', out_skip.shape)
    print('Output shape without skip net: ', out_noskip.shape)   

    print_gradients(skip_network, batch) 
    print_gradients(noskip_network,  batch) 


