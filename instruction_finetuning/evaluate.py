import os 
import sys
import urllib.request 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
import torch 
from model_architecture.gpt_model import GPTModel 
from attention.multihead_attention import ModelArgs 
import psutil 
import urllib
import json 

def check_if_running(process_name: str): 
    running = False 
    for proc in psutil.process_iter(["name"]): 
        if process_name in proc.info["name"]: 
            running = True 
            break 
    
    return running 

def query_ollama(prompt, model: str = 'llama3', url='http://localhost:11434/api/chat'): 
    data = {
        "model": model, 
        "messages": [
            {"role": "user", "content": prompt}
        ] , 

        "options": {
            "seed": 123, 
            "temperature": 0, 
            "num_ctx": 2048
        }
    } 

    payload = json.dumps(data).encode('utf-8') # Converts string dictionary to bytes since HTTP protocols operate with byte streams 
    request = urllib.request.Request(
        url = url, 
        data= payload, 
        headers = {'Content-Type': 'application/json'}, 
        method = 'POST' 
    ) 

    ollama3_response = "" 
    with urllib.request.urlopen(request) as response: 
        while True: 
            line = response.readline().decode('utf-8') 
            if not line: 
                break 
            response_json = json.loads(line) # line will also be a json dict string. Doing this gives us back the dictionary 
            ollama3_response += response_json["message"]["content"] 
    
    return ollama3_response 
 
if __name__ == '__main__': 
    
    # Using Ollam3 

    # Check if Ollama 3 is running 
    ollama_running = check_if_running("ollama") 
    if not ollama_running: 
        raise RuntimeError(
            "Ollama not running. Make sure to start ollama"
        )
    print("Ollama is running properly!") 

