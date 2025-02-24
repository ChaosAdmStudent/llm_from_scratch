### GPT Architecture

- A typical LLM architecture starts with Embedding layer for token and positions, adding them to make the input embeddings. The input embeddings are passed through Dropout layer.
- Next, the LLM is passed through many Transformer Blocks. A single Transformer block has a Masked Multi-Head Attention layer, followed by Layer normalization and a Feed Forward Layer (using GELU activation).
- The last output from the transformer block is passed through the last layer normalization layer again and then passed through a final feed forward network called the “Prediction Head”. This outputs the shape (B, num_tokens, vocab_size). The last dimension is  vocab size so that we can use the outputs to predict one of the words in the vocabulary.

### Layer Normalization

- Normalization is used in deep learning to prevent problem of vanishing/exploding gradients. This is achieved by making sure the activation values are always within a certain range.
- This makes sure that each layer’s input distribution is the same, which leads to faster convergence to optimum weights and stabilizes training. However, we don’t leave the mean and variance of input distribution to 1 and 0 respectively.
- We add a trainable scale and shift parameters which do affine transformation to the normalized values. The reason for doing this is, while we want to maintain stable range of values for the activations, we don’t want to force an input distribution on a layer. If a model so chooses that a different input distribution is the ideal way to minimize the loss function for that layer, having trainable scale and shift parameters allow the layer to operate in a different range of inputs.
    - Why do we still normalize then? The reason is that it gives the control to manipulate the input distribution to the layer itself. Not doing so would force an input distribution on the layer, while facing the problem of vanishing gradients.
    - By using scale and shift parameters, we don’t limit expressivity of the model.
    - For instance, if the next layer is using ReLU activation function, the scale and shift parameters for that layer’s LayerNorm may learn to not center the values around 0, but rather in a positive scale, so that negative values can contribute to the final outcome.

- In modern LLMs (emperically found), we use PreLayerNorm instead of original PostLayerNorm, i.e, we add the Layer Norm before the MultiHeadAttention and  FeedForward blocks inside the “Transformer Block”.

### GELU vs ReLU

- GELU is a more complex and smoother function than ReLU. Instead of hard-setting negative values to 0, it is a smoother function that allows some negative values to still contribute (albeit, much less than positive ones).
- The training becomes smoother and more stable with GELU.
- Actual GELU formula is $x.\phi(x)$  where $\phi(x)$ is the cumulative distribution function of Gaussian distribution.
- However, the above is too complex to compute; in practical scenarios, there are many approximations of the formula that are used instead.

### Skip Connections

- Skip connections are added after Dropout layers in the transformer blocks to help use shortcuts and help gradient flow back without going through a lot of layers. This can help significantly in gradient flow.
- Works complementarily with Layer Normalization which also helps with vanishing gradient problem.

### Generating Text from output head tensor

- Final gpt model output is of shape (B, num_tokens, vocab_size)
- The final generated values for each token is a tensor of size vocab_size, basically an unbounded logit value for each token.
- First, we convert logits into probabilities using softmax. Since, softmax is a monotonic function, i.e, it retains the order of the inputs, the argmax of logits and softmax will be the same. However, in the code I still do this to give model more intuition about the token selected. This is called “greedy decoding”
- We extract token id from the very last generated token’s probability scores. This is because the last generated output is the new word.
- Next, we add this generated output to the input tokens and return it.
- We end up returning a token iD which can be passed into the tokenizer’s decoder to get the corresponding text.