'''
In this file, I download the dataset of text with a "spam" or "not spam" classification. 
I prepare and split the data into train, test and val using pandas and then create PyTorch DataLoader classes for them. 
'''

from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import urllib.request 
import os 
from pathlib import Path
import zipfile
from pandas import DataFrame
import torch 
import tiktoken

def download_dataset(url, zip_path: Path, extracted_path: Path, file_path: Path): 
    
    if file_path.exists(): 
        print(f"Data already downloaded in {file_path}") 
        return 
    
    # Creates zip file from url 
    with urllib.request.urlopen(url) as response: 
        with open(zip_path, 'wb') as out_file: 
            out_file.write(response.read()) 
    
    # Unzips downloaded file 
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: 
        zip_ref.extractall(extracted_path) 
    
    # Rename file name to include .tsv extension
    original_file_path = Path(extracted_path)/"SMSSpamCollection"
    os.rename(original_file_path, file_path)
    print(f'Dataset downloaded and stored at {file_path}') 

def prepare_pd_dataset(data_file_path: Path): 
    """
    In this function, I undersample the dataset to have balanced class dataset 
    """

    df = pd.read_csv(data_file_path, sep='\t', header=None, names=['Label', 'Text']) 

    # Undersample 
    num_samples = df[df['Label'] == 'spam'].shape[0] 
    ham_subset = df[df['Label'] == 'ham'].sample(num_samples, random_state=123) 

    new_df = pd.concat((ham_subset, df[df['Label'] == 'spam'])).sample(frac=1.0, random_state=123).reset_index(drop=True)
    return new_df 

def split_train_test_val(df: DataFrame, train_frac=0.7, val_frac=0.1, test_frac=0.2): 
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(train_frac * len(df)) 
    val_end =  train_end + int(val_frac * len(df)) 
    df_train = df[:train_end] 
    df_val = df[train_end : val_end] 
    df_test = df[val_end: ] 

    return df_train, df_test, df_val

class SMSDataset(Dataset): 
    def __init__(self, df, tokenizer, max_length=None, pad_token=50256):
        if max_length is None: 
            self.max_length = len(max(df['Text'], key=len)) 
        else: 
            self.max_length = max_length

        # Fill sequence up to max_length with pad tokens 
        self.x = torch.full((len(df), self.max_length), fill_value=pad_token) # (num_samples, max_length) 
        self.y = torch.tensor(df['Label'].map({'ham': 0, 'spam': 1}).to_numpy()) # (num_samples,) 

        for i, text in enumerate(df['Text']): 
            tokens = torch.tensor(tokenizer.encode(text)) 
            self.x[i, :len(tokens)] = tokens 

    def __getitem__(self, index):
        return self.x[index], self.y[index] 

    def __len__(self): 
        return self.y.shape[0]
    
def create_data_loader(df, tokenizer, max_length, pad_token, batch_size, shuffle, drop_last): 
    dataset = SMSDataset(df, tokenizer, max_length, pad_token) 
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last
    )

if __name__ == '__main__': 
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv" 
    tokenizer = tiktoken.get_encoding('gpt2')
    pad_token = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    batch_size = 4 
    shuffle = True 
    drop_last = True 

    download_dataset(url, zip_path, extracted_path, data_file_path) 
    df = prepare_pd_dataset(data_file_path)    
    df_train, df_test, df_val = split_train_test_val(df)  

    train_loader = create_data_loader(df_train, 
                                      tokenizer, 
                                      max_length=None, 
                                      pad_token=pad_token, 
                                      batch_size=batch_size, 
                                      shuffle=shuffle, 
                                      drop_last=drop_last) 

    for x,y in train_loader: 
        print('X:', x.shape) 
        print('y: ', y.shape) 

        print(f'Decoded x: {tokenizer.decode(x[0].tolist())}') 
        print(f'Decoded y: {'ham' if y[0]=='0' else 'spam'}') 
        break 

    # Works!! 