import tiktoken

if __name__ == '__main__': 

    tokenizer = tiktoken.get_encoding('gpt2') 
    
    text = "Akwirw ier"

    ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'}) 
    print(ids) 
    subwords = [tokenizer.decode([i]) for i in ids] 
    print(subwords)
    decoded = tokenizer.decode(ids)
    print(decoded) 