## Overall process

These are the typical stages involved during pretraining an LLM: 

1. Text generation from model output 
2. Output text evaluation (loss function) 
3. Training loop (backprop to update params) 
4. Text generation strategies (To help model be more expressive/creative with responses) 
5. Using open weights (optional) 

## Text Generation

- To reiterate what was done in ch2, our gpt model predicts as many tokens in the output as input. The idea is to predict the next word for each token in the input. Thus, the input-output pairs are just shifted ahead by 1. The biggest benefit is that from one training example, we are essentially learning content_length number of input-output pairings instead of 1.

## Text Evaluation

- Use the Cross Entropy loss, since we are predicting next word based on classification (out of the vocabulary of size vocab_size.
- If implementing cross entropy manually, we have to basically take out the logit score of the vocab token which corresponds to the target token ID for that specific token position in the output. For example, if target token ids are: [1,4,5] for the 1st batch, this means that there were N = 3 tokens in that batch and 1,4, and 5 are  the tokens that should be predicted. 

Thus, we extract the logit scores at those indices in the $(N, vocab_{size})$ model output of the first batch.
- Since there’s only 1 correct word which should have target probability of 1 and all others must be 0, we can just calculate CE loss by: 
 
    $loss = -\sum_ilog(p_i)$

- Another text evaluation metric called “Perplexity” is often used with cross entropy loss. It is equal to torch.exp(ce_loss). 

It gives an interpretable number that tells you how many out of the N vocab words it is unsure to pick for the next word.  The lower this number, the more certain the model is about the next word.

## Decoding Strategies for variety

### Temperature Scaling 

- Temperature Scaling is a technique used in conjunction with a probabilistic sampling method to control the probability distribution of logits output by the LLM model to vary the diversity of the next token word selected during decoding.
- In greedy decoding, we always pick the token with max probability score. However, the problem with this is that the model will always give the same output text for a given input context. This is not good  for diversity responses.
- Thus, during evaluation when we are decoding, we can use a probabilistic sampling which samples based on the probability scores of the logits.
- While this will give diversity, if we want to control the level at which we want the model to alter the probability distributions, we use a concept called “Temperature Scaling”.
- The idea is to basically divide the logit scores by a number greater than 0.
    - Dividing by number greater than 0 and less than 1, will scale up logits. Since softmax uses exp function, the difference between logits is much larger if the logits are scaled up. Thus, this leads to a peak distribution, where logits which were originally large, are suddenly given much larger probability scores. 
    
    Hence, dividing by a number less than 1 results in more confident/less diverse model outputs
    - Similarly, scaling down reduces the gap between e^x terms, (x is the logit here). Thus, the higher the number is than 1, the more uniform the probability distributions become, and the more random the outputs start looking.

### Top-K sampling

- TopK sampling is used in conjunction with temperature scaling and probabilistic sampling. The idea is to sample logits from top-K probabilities instead of all the probabilities.
- TopK gives another level of control on the model output variety. In cases where we want to restrict the number of outputs, we should go for a lower TopK number.

## Other Learnings

- Other than the above, I mostly spent time writing the code for generating tokens, setting up the training loop, saving weights and loading open AI weights.
- A key part was also checking how to validate weights while loading into a GPT architecture.
- I took a long time to diagnose a problem with my GPT model, where the pretrained weights were giving really weird text outputs. After a long time of thinking, I found that my formula for calculating attention weights in MultiHeadAttention module was wrong. I was scaling down the attention scores by $d_{context\_dim}$ instead of $d_{head\_dim}$. Even though this is just a scalar that helps to bring down large values and thereby prevent smaller gradients, I think either would probably work, but the weights need to be trained for that scalar value.
    
    The Open AI GPT model was trained with division by $d_{head\_dim}$. Changing that fixed everything in the output that I was seeing and it became coherent.

- A big learning I had in the process came from implementing KV cache from scratch. 
    - Inference times were not as expected. I realized that I was testing with my own trained model with a very small context length and that GPU compute time at this small scale is overshadowed by overheads from python, cuda kernel launch etc. Thus, I switched to using the model with OpenAI weights and the inference times reflected the trend much better. 

    - My outputs were not matching when compared to non KV mode. After countless hours, I did the following: 
        - Printed shapes of different intermediate variables in the entire MHA architecture. This was not a problem. 
        - Added debugging functionality to forward() functions of MHA to retrieve Q,K and V at different stages of computation in KV and non KV mode to compare values. This showed that initial prompt commputations were matching but as soon as I was processing the new token directly through the architecture, the Q,K and V of the new token did not correspond with the last token's in non KV mode. 

        - The above made me question computation outside of the MHA architecture itself. I realized that the problem was with positional embedding. I had `pos_embedding(torch.arange(num_tokens))` which works fine for non KV mode when we have all the tokens in the sequence. But for KV mode, this meant I was always adding the PE of the 0th index (1st position) to the input token embedding. This was the main reason the outputs looked gibberish with KV mode. I fixed it by using `pos_embedding(torch.tensor([start_pos]))` where start_pos maintained the current position of the word in the generated sequence. This fixed everything and I could reproduce the same results in both KV and non KV mode. 