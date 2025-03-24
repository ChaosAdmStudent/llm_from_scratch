## Prompt styles 

- We can format the dataset of {instruction, input, output} in different formats. Like: 
    - Alpaca prompt style: 
        - Below is an instruction that describes a task. Write a respoonose that appropriately completes the request. 
        
            \### Instruction: 

            Identify the correct spelling of the following word.

            \### Input: 

            Ocasion 

            \### Response: 

            The correct spelling is 'Occasion' 
    
    - Phi-3 prompt style: 
        - <|user|> 

            Identify the correct spelling of the following word: 'Ocassion' 

            <|assisstant|>

            The correct spelling is 'Occasion'. 

- Can different prompt styles lead to different performances? 
    - *TBD* 

## Step by step process for organizing data into training batches 

1. Format data using format template 
2. Tokenize formatted data 
3. Adjust to the same length with padding tokens 
4. Create target IDs by shifting input to the left by 1 and inserting end-of-text token at the end. Similar to pretraining, we want the model to learn to predict next token using the input itself. 
5. Replace padding tokens with placeholders in target tensor. We will use this placeholder value to help training loop ignore these values so that model is not influenced by padded values. PyTorch's cross entropy loss function has an `ignore_index` parameter that can be used to skip computation over these values. 

6. (Optional) We can also mask out tokens inside target tensor corresponding to input and instruction data. That way, model will only learn to generate the responses and not learn the instructions themselves, thereby leading to less overfitting. But, some papers have shown that doing so can reduce model performance, so it's something that needs to be tested out. 

## Custom collate functions for preparing PyTorch dataloader with variable input sizes

- We can create a custom collate function that can be passed as an argument to PyTorch's DataLoader class. This function is responsible for gathering different elements returned by Dataset's `__getitem__` function. In our case, that function returns a list of tokens corresponding to a prompt. So for a batch size of 4, a given batch will get 4 such lists combined together in one list. This combined list is what the `collate_fn(batch)` gets. We can decide how we process the batch and return it. 

- Since the inputs and outputs can have different lengths, the prepared tokenized formatted prompts will have different lengths. We will pad all the formatted prompt token lists till the length of the largest list. I pad with 50256 (corresponding to "<|endoftext|>")

- Then we will replace the padded tokens with -100, i.e, a negative number so that loss function can later be told to ignore negative values. 

## Why use a collate function here but not for classification fine-tuning? 

- Well, we could have done static padding like we did in classification fine-tuning. There, we created the padded input directly inside the Dataset class, i.e, we padded to a given max_length before hand. 

- Results wise, the above would achieve the exact same output. However, it's the usual computer science tradeoff between runtime and memory. 

    ### Static Padding (Padding inside init function of Dataset)
    - All sequences are padded to the same global max length in the dataset’s `__init__` 
    - **Memory usage**: Potentially large if your dataset’s max length is big (you store a large 2D tensor right away). 
    - **Runtime**: When fetching a sample, there is no extra padding cost at dataloader time since input is already prepared with correct shape
    - **Less flexibility**: You cannot easily adapt the max length for each batch or run-time conditions.

    ### Dynamic Padding (Padding using collate function passed to Dataloader) 
    - The dataset returns variable-length items.
    - The custom collate_fn does the padding at each batch creation step.
    - **Memory usage**: More efficient if you do “dynamic padding” – i.e., pad each batch only up to its largest item, not the entire dataset’s largest. 
    - **Runtime**: Slight overhead at each batch to find the local max length and pad, but less memory wasted.
    - **More flexible**: You can do bucketing or adapt max lengths on a per-batch basis. This is commonly used in NLP tasks. 

- In practice, since the max_length is very high, we don't want to create and keep a global matrix with that many columns in memory. Thus, LLM and NLP applications generally use a custom collate_fn instead to handle variable sized inputs.

## Moving input and output to device inside collate function instead of training loop

- This is helpful because it helps moving tensors to the GPU when pre-fetching the batch during training. If we have num_workers>0, the tensors can be already moved to the GPU while the training loop is executing. This saves time during training, especially when working with huge tensors. 

- However, to use num_workers>0 and transferring cpu tensors to gpu inside collate function, we can't simply do that. We need to use mp.spawn because CUDA does not support forking and there will be an error otherwise saying that `Cannot reinitialize CUDA in forked subprocess` 