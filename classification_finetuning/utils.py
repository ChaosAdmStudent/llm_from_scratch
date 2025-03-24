import matplotlib.pyplot as plt 
import os 
import random 
import numpy as np 
import torch 

def plot_values(epochs_seen, examples_seen, train_values, val_values, label:str): 
    fig, ax1 = plt.subplots(figsize=(5,3)) 

    # Plot Train values 
    ax1.plot(epochs_seen, train_values, label=f"Training {label}") 
    ax1.plot(epochs_seen, val_values, label=f"Validation {label}", linestyle='-.') 
    ax1.set_xlabel("Epochs seen") 
    ax1.set_ylabel(label.capitalize()) 
    ax1.legend()

    # Plot Val values
    ax2 = ax1.twiny() 
    ax2.plot(examples_seen, train_values, alpha=0) # Invisible/transparent plot for aligning ticks for 2 X-axis
    ax2.set_xlabel("Examples seen") 

    fig.tight_layout() 

    plot_dir = 'classification_finetuning/plots'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/model_{label}.png', bbox_inches='tight') 

def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True