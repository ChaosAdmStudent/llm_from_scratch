'''
This file has code to run inference on my trained model. To use openAI weights to do inference, use pretrained_openai.py. 

TODO: Add padding functionality for multi-batched inputs. All code assumes equal number of tokens for each batch right now.
'''

import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import tiktoken 
from ch5.pretraining import GPT_CONFIG_124M 
from ch5.utils import generate
from ch4.gpt_model import GPTModel
from ch3.multihead_attention import ModelArgs 
import torch 
from time import perf_counter
import numpy as np 
import matplotlib.pyplot as plt 
import gc

class Timer(): 
    def __init__(self):
        self.duration = []

    def __start__(self): 
        torch.cuda.synchronize() 
        self.start = perf_counter()  

    def __end__(self): 
        torch.cuda.synchronize() 
        self.duration.append(perf_counter() - self.start)  

    def __reset__(self): 
        self.duration = []  

    def __duration__(self): 
        return np.mean(self.duration)  

'''
This file has code to run inference on my trained model. To use openAI weights to do inference, use pretrained_openai.py. 

TODO: Add padding functionality for multi-batched inputs. All code assumes equal number of tokens for each batch right now.
'''

import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import tiktoken 
from ch5.pretraining import GPT_CONFIG_124M 
from ch5.utils import generate
from ch4.gpt_model import GPTModel
from ch3.multihead_attention import ModelArgs 
import torch 
from time import perf_counter
import numpy as np 
import matplotlib.pyplot as plt 
import gc

class Timer(): 
    def __init__(self):
        self.duration =  []

    def __start__(self): 
        torch.cuda.synchronize() 
        self.start = perf_counter()  

    def __end__(self): 
        torch.cuda.synchronize() 
        self.duration.append(perf_counter() - self.start) 

    def __reset__(self): 
        self.duration = [] 

    def __duration__(self): 
        return np.mean(self.duration) 

def compare_average_inference_time(model, input_tokens, max_new_tokens, context_length, device, temperature, num_runs=10): 
    
    timer = Timer() 

    # Without KV Cache
    model.toggle_kv_cache(False) 
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(num_runs): 
            timer.__start__() 
            generated_tokens = generate(max_new_tokens, model, input_tokens, context_length, device, temperature=temperature, use_kv_cache=False) 
            timer.__end__() 

    time2 =  timer.__duration__()
    
    # With KV Cache
    model.toggle_kv_cache(True) 
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(num_runs): 
            timer.__start__() 
            generated_tokens = generate(max_new_tokens, model, input_tokens, context_length, device, temperature=temperature, use_kv_cache=True) 
            timer.__end__() 

    time1 =  timer.__duration__() 

    return time1, time2  

def plot_inference_times(seq_lens, times_kv, times_no_kv, out_folder):
    plt.figure(figsize=(10, 6))
    
    # Plot lines with markers
    plt.plot(seq_lens, times_kv, marker='o', label='With KV Cache', color='blue')
    plt.plot(seq_lens, times_no_kv, marker='s', label='Without KV Cache', color='red')
    
    # Add labels and title
    plt.xlabel('Input Sequence Length', fontsize=12)
    plt.ylabel('Average Inference Time (seconds)', fontsize=12)
    plt.title('KV Cache Performance Comparison', fontsize=14)
    
    # Use logarithmic scale if sequence lengths vary widely
    # if max(seq_lens)/min(seq_lens) > 10:
    #     plt.xscale('log')
    #     plt.yscale('log')
    
    # Add grid and legend
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Annotate crossover point
    crossover = None
    for i, (kv, nokv) in enumerate(zip(times_kv, times_no_kv)):
        if kv < nokv and crossover is None:
            plt.annotate('KV Cache Becomes Faster', 
                         (seq_lens[i], kv),
                         textcoords="offset points",
                         xytext=(-20,20),
                         arrowprops=dict(arrowstyle="->")) 
            break
    
    plt.tight_layout()
    os.makedirs(out_folder, exist_ok=True)
    plt.savefig(f'{out_folder}/compare_time_kv_nokv.png', bbox_inches='tight') 

# Improved testing function
def run_benchmarks(model, cfg, text_prompts,  tokenizer, device, max_new_tokens, out_folder):
    
    # Initialize containers
    seq_lens = []
    kv_times = []
    no_kv_times = []
    
    for i,prompt in enumerate(text_prompts):
        print(f'Processing prompt {i}')
        # Tokenize and process
        input_tokens = torch.tensor([tokenizer.encode(prompt)]).to(device).repeat(4, 1)
        seq_len = input_tokens.shape[1]
        
        # Warmup runs (avoid cold start measurements like one time cuda kernel launch latency)
        _ = compare_average_inference_time(model, input_tokens, max_new_tokens, 
                                         cfg["context_length"], device, 
                                         temperature=1, num_runs=2)
        
        print('\t Warmup run completed') 
        # Actual measurement
        time_kv, time_no_kv = compare_average_inference_time(
            model, input_tokens, max_new_tokens, 
            cfg["context_length"], device, temperature=1, num_runs=6
        )
        print('\t Profiling completed')
        
        seq_lens.append(seq_len)
        kv_times.append(time_kv)
        no_kv_times.append(time_no_kv)

        del input_tokens  
        torch.cuda.empty_cache() 
        gc.collect() 

    
    # Sort results by sequence length
    sorted_indices = np.argsort(seq_lens)
    seq_lens = np.array(seq_lens)[sorted_indices]
    kv_times = np.array(kv_times)[sorted_indices]
    no_kv_times = np.array(no_kv_times)[sorted_indices]
    
    # Generate plot
    plot_inference_times(seq_lens, kv_times, no_kv_times, out_folder)
    
    return seq_lens, kv_times, no_kv_times


if __name__ == '__main__': 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    tokenizer = tiktoken.get_encoding('gpt2') 
    max_new_tokens = 300
    chk_path = 'ch5/model_checkpoint.pth' 
    out_folder = 'ch5/plots'

    text_prompts = [
        "I am",  # Very short
        "I am going to be a really good person.",  # Medium
        "I am going to be a really good person. I don't know how that feels like but",  # Long
        "I am going to be a really good person. I don't know how that feels like but there is just something that I know deep down that I can go for it if I",  # Very long
        "The error likely stems from a mismatch in the GPU type specification. Based on the node's GRES (Generic Resources) configuration, here's how to adjust your script I always said that God makes the answer in everything, now you got my answer in musical notes, I feel as if she brings peace to me, she tells me that what my heart wants is right, even though I'm surrounded by a pathetic reality, but I feel like something"
    ]

    checkpoint = torch.load(chk_path, map_location=device)
    kv_args = ModelArgs()
    model = GPTModel(GPT_CONFIG_124M, kv_args) 
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Compare inference times with and without KV cache for different sequence lengths 
    seq_lens, kv_times, no_kv_times = run_benchmarks(model, GPT_CONFIG_124M , text_prompts, tokenizer, device, max_new_tokens, out_folder) 
    
    print(seq_lens) 
    print(kv_times)
    print(no_kv_times) 
