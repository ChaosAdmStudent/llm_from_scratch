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

def profile_generate(model, input_tokens, max_new_tokens, context_length, device, use_kv_cache, num_runs=3):
    """
    Runs inference with and without KV cache while collecting detailed profiler information.
    """
    timer = Timer()
    model.toggle_kv_cache(use_kv_cache)

    gc.collect()
    torch.cuda.empty_cache() 

    total_cuda_time = 0

    for _ in range(num_runs): 
        # Initialize PyTorch profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False
        ) as prof:
            timer.__start__()
            _ = generate(max_new_tokens, model, input_tokens, context_length, device, temperature=1.0, use_kv_cache=use_kv_cache)
            timer.__end__() 
            total_cuda_time += sum(evt.device_time for evt in prof.key_averages())

        torch.cuda.empty_cache()
        gc.collect()
    
    avg_time = timer.__duration__()
    del prof 
    
    return avg_time, total_cuda_time

def compare_average_inference_time(model, input_tokens, max_new_tokens, context_length, device, num_runs=3):
    """
    Compares inference time and CUDA kernel time with and without KV cache.
    """

    gc.collect()
    torch.cuda.empty_cache()

    time_kv, cuda_kv = profile_generate(model, input_tokens, max_new_tokens, context_length, device, use_kv_cache=True, num_runs=num_runs)
    time_no_kv, cuda_no_kv = profile_generate(model, input_tokens, max_new_tokens, context_length, device, use_kv_cache=False, num_runs=num_runs)

    return time_kv, time_no_kv, cuda_kv, cuda_no_kv

def plot_inference_times(seq_lens, times_kv, times_no_kv, cuda_times_kv, cuda_times_no_kv, out_folder):
    plt.figure(figsize=(10, 6))

    # First plot: total inference time
    plt.subplot(1, 2, 1)
    plt.plot(seq_lens, times_kv, marker='o', label='With KV Cache', color='blue')
    plt.plot(seq_lens, times_no_kv, marker='s', label='Without KV Cache', color='red')
    plt.xlabel('Input Sequence Length')
    plt.ylabel('Average Inference Time (seconds)')
    plt.title('KV Cache Performance Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add only **one arrow** at the first point where KV cache is faster
    for i in range(len(seq_lens)):
        if times_kv[i] < times_no_kv[i]:
            plt.annotate('KV Cache Becomes Faster', 
                         (seq_lens[i], times_kv[i]),
                         textcoords="offset points",
                         xytext=(-20, 20),
                         arrowprops=dict(arrowstyle="->"))
            break  # Stop after the first annotation

    # Second plot: CUDA time comparison
    plt.subplot(1, 2, 2)
    plt.plot(seq_lens, cuda_times_kv, marker='o', label='With KV Cache (CUDA time)', color='blue')
    plt.plot(seq_lens, cuda_times_no_kv, marker='s', label='Without KV Cache (CUDA time)', color='red')
    plt.xlabel('Input Sequence Length')
    plt.ylabel('Total CUDA Execution Time (ms)')
    plt.title('CUDA Time for KV vs Non-KV')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    os.makedirs(out_folder, exist_ok=True)
    plt.savefig(f'{out_folder}/compare_kv_cache.png', bbox_inches='tight')

def run_benchmarks(model, cfg, text_prompts, tokenizer, device, max_new_tokens, out_folder):
    seq_lens, kv_times, no_kv_times, cuda_kv_times, cuda_no_kv_times = [], [], [], [], []

    for i,prompt in enumerate(text_prompts):
        print(f'Processing prompt {i}') 
        input_tokens = torch.tensor([tokenizer.encode(prompt)]).to(device)
        seq_len = input_tokens.shape[1]

        # Warm-up run (to avoid cold-start bias)
        _ = compare_average_inference_time(model, input_tokens, max_new_tokens, cfg["context_length"], device, num_runs=1)

        print('\t Warmup run completed')

        # Actual measurements
        time_kv, time_no_kv, cuda_kv, cuda_no_kv = compare_average_inference_time(
            model, input_tokens, max_new_tokens, cfg["context_length"], device, num_runs=3
        )

        print('\t Profiling completed')

        seq_lens.append(seq_len)
        kv_times.append(time_kv)
        no_kv_times.append(time_no_kv)
        cuda_kv_times.append(cuda_kv)
        cuda_no_kv_times.append(cuda_no_kv)

        del input_tokens
        torch.cuda.empty_cache()
        gc.collect()

    # Sort results for better plotting
    sorted_indices = np.argsort(seq_lens)
    seq_lens = np.array(seq_lens)[sorted_indices]
    kv_times = np.array(kv_times)[sorted_indices]
    no_kv_times = np.array(no_kv_times)[sorted_indices]
    cuda_kv_times = np.array(cuda_kv_times)[sorted_indices]
    cuda_no_kv_times = np.array(cuda_no_kv_times)[sorted_indices]

    plot_inference_times(seq_lens, kv_times, no_kv_times, cuda_kv_times, cuda_no_kv_times, out_folder)

    return seq_lens, kv_times, no_kv_times, cuda_kv_times, cuda_no_kv_times

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
    seq_lens, kv_times, no_kv_times, cuda_kv_times, cuda_no_kv_times = run_benchmarks(model, GPT_CONFIG_124M , text_prompts, tokenizer, device, max_new_tokens, out_folder) 
    
    print(seq_lens) 
    print(kv_times)
    print(no_kv_times) 
