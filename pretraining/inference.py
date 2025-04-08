'''
This file has code to run inference on my trained model. To use openAI weights to do inference, use pretrained_openai.py. 

TODO: Add padding functionality for multi-batched inputs. All code assumes equal number of tokens for each batch right now.
'''

import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import tiktoken 
from pretraining.pretraining import GPT_CONFIG_124M 
from pretraining.utils import generate
from model_architecture.gpt_model import GPTModel
from attention.multihead_attention import ModelArgs 
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
