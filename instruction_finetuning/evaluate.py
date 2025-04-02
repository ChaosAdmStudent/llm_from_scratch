import urllib.request 
import psutil 
import urllib
import json 
from tqdm import tqdm 
import sys 
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from instruction_finetuning.data_prep import format_input_alpaca
from pathlib import Path
import tyro
from openai import OpenAI
from instruction_finetuning.utils import run_chatgpt

def check_if_running(process_name: str): 
    running = False 
    for proc in psutil.process_iter(["name"]): 
        if process_name in proc.info["name"]: 
            running = True 
            break 
    
    return running 

def query_ollama(prompt, model: str = 'phi3', url='http://localhost:11434/api/chat'): 
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

def generate_model_scores(json_data, json_key, api = 'ollama', model: str = 'phi3', client = None): 
    """
    Generates score from 0 to 100 for my fine-tuned model output.
    json_data: json file data where model outputs on test data have been stored. 
    json_key: The key corresponding to my model response in json data
    """
    
    scores = [] 
    for entry in tqdm(json_data, total=len(json_data)): 
        prompt = (
            f"Given the input `{format_input_alpaca(entry)}` " 
            f"and correct output: `{entry['output']}`" 
            f"score the model response: {entry[json_key]}" 
            "on a scale from 0 to 100, where 100 is the best score." 
            "Respond with integer number only" 
        ) 

        if api == 'ollama':
            eval_response = query_ollama(prompt, model, 'http://localhost:11434/api/chat') 
        elif api == 'openai': 
            assert client is not None, "Did not provide OpenAI client object"
            eval_response = run_chatgpt(prompt, client, model)

        try: 
            score = int(eval_response) 
            scores.append(score) 
        except: 
            print(f'{api} response: {eval_response}')
            print(f"{api} response not convertable to int")  

    return scores 

def main(eval_model: str = 'openai', model: str = 'gpt-4-turbo'): 
    # Load model responses JSON file 
    file_path = Path('instruction_finetuning/finetune-responses.json') 
    with open(file_path, 'r') as file: 
        json_data = json.load(file)

    if eval_model == 'openai': 
        # Load OpenAI client using API key
        cur_folder = os.path.dirname(os.path.abspath(__file__))
        with open(f'{cur_folder}/config.json') as file: 
            config = json.load(file) 
            api_key = config['OPENAI_API_KEY'] 

        client = OpenAI(api_key=api_key)   

        # Check if OpenAI client works 
        prompt = "say 'hello' if you can process" 
        gpt_output = run_chatgpt(prompt, client) 
        print(gpt_output)  

        # Use ChatGPT for evaluation 
        evaluation_scores = generate_model_scores(json_data, 'model_response', api='openai', model='gpt-4-turbo', client=client)

    else: 
        # Using Ollama3 
        # Check if Ollama 3 is running 
        ollama_running = check_if_running("ollama") 
        assert ollama_running,  "Ollama not running. Make sure to start ollama"
        print("Ollama is running properly!")   
        
        model = 'phi3' 
        evaluation_scores = generate_model_scores(json_data, 'model_response', api='ollama', model='phi3') 
    
    print(f'Average score with {model} model: {sum(evaluation_scores)/len(evaluation_scores):.2f}') 
 
if __name__ == '__main__': 
    tyro.cli(main) 

    
