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

- Temperature Scaling is a technique used in conjunction with a probabilistic sampling method to control the probability distribution of logits output by the LLM model to vary the diversity of the next token word selected during decoding.
- In greedy decoding, we always pick the token with max probability score. However, the problem with this is that the model will always give the same output text for a given input context. This is not good  for diversity responses.
- Thus, during evaluation when we are decoding, we can use a probabilistic sampling which samples based on the probability scores of the logits.
- While this will give diversity, if we want to control the level at which we want the model to alter the probability distributions, we use a concept called “Temperature Scaling”.
- The idea is to basically divide the logit scores by a number greater than 0.
    - Dividing by number greater than 0 and less than 1, will scale up logits. Since softmax uses exp function, the difference between logits is much larger if the logits are scaled up. Thus, this leads to a peak distribution, where logits which were originally large, are suddenly given much larger probability scores. 
    
    Hence, dividing by a number less than 1 results in more confident/less diverse model outputs
    - Similarly, scaling down reduces the gap between e^x terms, (x is the logit here). Thus, the higher the number is than 1, the more uniform the probability distributions become, and the more random the outputs start looking.