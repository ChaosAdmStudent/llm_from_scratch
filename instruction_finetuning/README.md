# Data Preparation for Instruction Fine-tuning 

Data preparation is a big part of instruction fine-tuning. It involves creating carefully designed prompt templates including an instruction, input (optional) and an output. There are many different styles of doing so:

### Alpaca Formatting 

```
"Below is an instruction that describes a task. Write a response that appropriately completes the request." 

### Instruction  
<some instruction> 

### Input  
<some input> 

### Output  
<corresponding output>
``` 

### Phi-3 Formatting 

```
<|user|>
<some instruction (+input)> 

<|assisstant|> 
<corresponding output>
``` 

# Handling batches of different sizes 
Since each training sample will have a different token embedding sequence length, it is important to handle varying sequences. There are two ways of doing this and both have its pros and cons. It's more common to go with dynamic padding. 

### Static Padding (Padding inside Dataset class)
- All sequences are padded to the same global max length in the dataset’s __init__.
- Memory usage: Potentially large if dataset’s max length is big (you store a large 2D tensor right away).
- Runtime: When fetching a sample, there is no extra padding cost at dataloader time. 
- Less flexibility: You cannot easily adapt the max length for each batch or run-time conditions. 

### Dynamic Padding (Padding inside Collate function)
- The dataset returns variable-length items; the custom collate_fn does padding at each batch creation step. 
- Memory usage: More efficient if you do “dynamic padding” – i.e., pad each batch only up to its largest item, not the entire dataset’s largest. 
- Runtime: Slight overhead at each batch to find the local max length and pad, but less memory wasted. 
- More flexible: You can do bucketing or adapt max lengths on a per-batch basis. This is commonly used in NLP tasks. 

We also ensure that all the padded tokens except one in the output tensor is masked away by some value like -100. This value is then passed to the `ignore_index` argument in the loss function indicating that the loss should not be influenced by padded tokens. The one token we retain is for indicating end-of-sequence. We want the model to learn when to stop generating more text. 

# Fine-tuning training and masking instruction tokens in output tensor
The actual fine-tuning training is similar to pretraining. The only difference is that this time we are doing next-word prediction on the prepared prompt which also has the output encapsulated in it. The one thing that is different (optional) is that we can also mask away the output tokens corresponding to the instruction itself. This can help the model focus on learning the outputs, instead of learning to just regenerate the provided input. Since pretraining has already been completed before this, we don't necessarily need to make the model learn the grammatical structure of the text, which is the only real learning coming from learning to generate the same. However, this is not a hard and fast rule and there are research papers that have proved benefits of not doing this. 

# Evaluating fine-tuned model outputs 

Since the output of the fine-tuned LLM is a generated text, it's not as simple to evaluate as classification fine-tuning where statistical metrics like accuracy can be used. Evaluating LLMs can be done with both statistical measures and model-based scorers. Statistical scorers don't take into consideration any semantic relationships in the output which is very important for language based assessments. It's more common to use model-scorers or a combination of the two. 

Model scorers involve using another LLM to judge and provide a score on the model response compared to the corresponding output. 
