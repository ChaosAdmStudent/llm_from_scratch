import re 

class SimplerTokenizer(): 

    '''
    A simple tokenizer that does not use special context tokens. 
    ''' 
    

    def __init__(self, text): 
        tokenized = self.tokenize(text) 
        self.tk_to_ids = {tk:i for i,tk in enumerate(sorted(set(tokenized)))} 
        self.ids_to_tk = {i:tk for tk,i in self.tk_to_ids.items()}  

    def tokenize(self, text): 
        tokenized = re.split('([,.:;?_!()"\']|--|\s)', text) 
        tokenized = [token for token in tokenized if token.strip()]  
        return  tokenized 

    def encode(self, text): 
        tokenized = self.tokenize(text) 
        return [self.tk_to_ids[token] for token in tokenized]  
    
    def decode(self, ids): 
        decoded = " ".join([self.ids_to_tk[i] for i in ids]) 
        decoded = re.sub(r'\s+([,.:;?_!()"\'])', r'\1', decoded) 
        return decoded 
    
class SimplerTokenizerV2(): 
    '''
    Tokenizer with special context tokens. 
    '''
    def __init__(self, text): 
        tokenized = self.tokenize(text) 
        self.tk_to_ids = {tk:i for i,tk in enumerate(sorted(set(tokenized)))} 
        self.tk_to_ids['<|unk|>'] = len(self.tk_to_ids) # Special token for unknown words 
        self.tk_to_ids['<|endoftext|>'] = len(self.tk_to_ids) # Special token for end of text to seperate indepdenent text sources 
        self.ids_to_tk = {i:tk for tk,i in self.tk_to_ids.items()}   

    def tokenize(self, text): 
        tokenized = re.split('([,.:;?_!()"\']|--|\s)', text) 
        tokenized = [token for token in tokenized if token.strip()]  
        return tokenized 

    def encode(self, text): 
        tokenized = self.tokenize(text) 
        return [self.tk_to_ids[token] if token in self.tk_to_ids.keys() else self.tk_to_ids['<|unk|>'] for token in tokenized]    
    
    def decode(self, ids): 
        decoded = " ".join([self.ids_to_tk[i] for i in ids])  
        decoded = re.sub(r'\s+([,.:;?_!()"\'])', r'\1', decoded) 
        return decoded 


if __name__ == '__main__': 
    # Load the text 
    with open('ch2/the-verdict.txt','r') as book: 
        text = book.read() 
    
    tokenizer = SimplerTokenizerV2(text)  
    sample_text1 = """Hey man, what's up? Lakshya here."""
    sample_text2 = "Cute is a nice word. I use it to describe my gf." 
    text = "<|endoftext|> ".join([sample_text1, sample_text2]) 

    print(text) 

    ids = tokenizer.encode(text) 
    print(ids) 

    decoded = tokenizer.decode(ids)
    print(decoded)

    
