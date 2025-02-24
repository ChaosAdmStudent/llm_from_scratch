### What is Self Attention and why we need it

- Self attention was made so that decoder can look at different states of the encoder or different words in the original input to give the predicted output.
- RNNs had a big problem of capturing all the context of the entire input sequence into one memory cell towards the end, which was sent to the decoder to generate output. This lead to loss of context, especially in longer sequences because of vanishing gradient problem.
- Self attention is a mechanism that helps every position in the sequence to consider the relevance of all other positions in the sequence to update each word’s, and consequently the entire sequence’s, vector representation (latent representation).
- Initial token embeddings coming out during data preparation is essentially a look up operation, where the same word will always give the same token embedding initially. However, that same word can have very different meanings depending on the context in which it was used.
- Self attention allows token embeddings of one word to modify the token embeddings of other words. It allows flow of information from one embedding to the other. You can think of it as enabling the generic token embedding vector to be shifted in the vector space in a direction that aligns with what that word actually means or conveys in the sentence. Think of the same word having many many actual vector directions in the latent space and self attention allowing the generic token embedding vector to be moved in the direction of one of those that captures the word’s meaning.
- Book goes through 4 types of self-attention to build up to an efficient implementation of multi-head attention at the end of the chapter.
    
    ### Basic internal working of Simple Self Attention
    
- Self attention works by computing a similarity score (also called attention weight) of each token with respect to all other tokens.
- The attention weights are used to calculate a weighted sum of the token embeddings to get a context vector for each “query” token. This context vector essentially captures all the enriched information about that “query” word after looking at all the other tokens in the sequence. This is what I meant by “able to modify the embedding of each word”. These context vectors the ones that do that.
    
    ### How are Context Vectors calculated?
    
- Let’s say we have initial token embeddings for $x^{(1)}, x^{(2)}...x^{(T)}$. If we want context vector for $x^{(2)}$, we can calculate it as the weighted sum of token embeddings weighted using each token’s attention weight (aka similarity score).
    
    That is, $z^{(2)} = \alpha_{21}*x^{(1)} + \alpha_{22}*x^{(2)} + ... \alpha_{2T}*x^{(T)}$  
    
- These attention scores are calculated using a dot product of the “query” embedding token (here that’s $x^{(2)}$ with all other embedding tokens. This initially gives us intermediate attention weights $\omega$.
- We get the final  attention weights $\alpha$ by NORMALIZING the intermediate attention weights. This normalization is done so that:
    - All attention weights sum up to 1.
    - All attention weights have positive values (This is done using softmax)
    - The attention weight values are interpretable and stabilize LLM training.
    - Attention weights become interpretable as we can look a them like probabilities, i.e, the higher the attention weight, the higher a specific token had impact on the “query” token.

### Self Attention with trainable weights

- Just like above, the focus here is still to produce context vectors that are produced by a weighted sum of the input vectors specific to the corresponding input elements.
- The big difference is that, instead of looking at the initial token embeddings itself as the “input vectors” in the computation, we instead project the input embedding of each token into 3 weight matrices: $W_Q, W_K, W_V$ .
- The idea is that we will perform calculations on the “projected vectors” for each token embedding. The projected vectors for one token are calculated by taking the matrix multiplication of the 3 weight matrices with the input embedding of that token.
- The weights of these matrices are learnable, and that gives the model a chance to learn “GOOD” context vectors by consequently altering the input vectors that are learned from a token embedding.
- So, if we want the context vector for $x^{(2)}$ , we will:
    - Compute attention scores for all tokens with respect to $x^{(2)}$  by finding the dot product of the “Query” projection of $x^{(2)}$  with the “Key” projections of all other tokens.
    - Dividing/Scaling the attention scores by $\sqrt{d_k}$ where $d_k$ is the number of projection dimensions. (Usually this is equal to dimensionality of token embedding). This scaling is done to scale down large dot product values, which are typical in LLMs operating at very high dimensions. This helps with stable gradients because large values inside softmax leads to very small gradients. Not scaling the scores will lead to gradient updates close to 0 and the model won’t learn anything.
    - Normalizing the attention scores using softmax
    - Using the attention scores as weights and then computing weighted sum over the “Values” projection vectors of all tokens.
- In essence, previously we were using the input token embeddings itself to calculate both attention as well as the weighted sum. Now, we are using 2 separate projection vectors to get similarity scores (Queries and Keys) and then a 3rd projection vector to carry out the weighted sum (Values).
- The Q,K,V terminology comes from database lingo.
    - Query is similar to search query in a database. For LLMs, this means the token that the model is trying to understand.
    - Keys in databases are like indexes which are used to match a query provided by the user. In LLMs, this means each item in the sequence has an associated key to match the query.
    - Values in databases are the actual content stored in a typical key-value pair. In LLMs, this implies the actual representation/content of the item  in the sequence. Once the model figures out which key matches the query in the best way (Attention scores), it retrieves the corresponding values. This is why we use the “values” projection vector to do the actual weighted sum over to get the context vectors. The context vectors are composed of data coming from the “values” projection as we are kind of explicitly telling the model that the actual representation/data about the item is in this vector.
- An intuitive example is if the self attention decided that for one of the heads it is only tasked with understanding how the adjectives update a noun, then in that case for the noun words, these would be the Q, K and V:
    - Query: “Are there any adjectives before me?”
    - Key: That word’s Part of speech. Basically a vector that identifies whether a word is a noun, pronoun, adjective etc.
    - Value: the actual values.
    
    So for a sentence the blue fluffy creature”, if we compute context vector for “creature”, the query is the question stated above. The keys for token 2 and 3 are: adjective.  The values are the actual content stored in those token positions, aka, “the” “blue” and “fluffy”. Hence for this head, the similarity scores for the given query will be much higher for token 2 and 3 and they would contribute most to the context vector of creature. 
    

- Using $nn.Linear$ layers instead of $nn.Parameter$ is more efficient for Self Attention block because:
    - nn.Linear has more effective matrix multiplication done in the backend.
    - nn.Linear has a more optimized weight initialization strategy leading to stable gradients and better model training.

### Why do we project input token embeddings into Q,K and V?

- Projecting the input token embeddings into Q, K and V have the following main benefits:
    - Interpretability. Since Q and K are tasked with similarity and V with actual content representation.
    - Decoupling tasks allow model to be more expressive.

### What is Causal Self Attention?

- Masking out future token embeddings’ weights so that current word’s context vector is only informed by current and previous inputs. This is important for certain LLM tasks like language modelling where we are predicting next word based on previous ones. This is typically used in the decoder.

### Using Dropout in attention weights?

- Dropout is used in deep learning to prevent overfitting by randomly ignoring/dropping out some hidden layer units to make the model less reliant on specific hidden layer units.
- Dropout only done during training, not testing
- When performing Dropout operation, we also scale up the retained weights by $\dfrac{1}{1-p}$ so that the expected value of the attention weights $a_i$ remains the same during training and inference.
- This scaling is crucial because without it, the model would learn smaller attention values during training, i.e, attention weights not adding up to 1, but during testing it will face larger attention values, leading to poor generalization results because model has learnt to operate on different expected  attention values.

### What is Multi-Head Attention

- Multi-Head Attention involves creating multiple “heads” by projecting the input token dimensions into multiple sets of Q,K and V. The self attention operations for different sets of Q,K and V are computed in parallel.
- The main reason of using multi-head attention is that it gives the model a chance to learn different aspects/relationships between the tokens. For instance, one head could focus on the syntactical relationships between tokens while another head could focus on the semantic relationships. It could also mean that one of the heads is focusing on understanding how the NOUNS are changed by adjectives while other heads focus on other parts of speech.

### Practically using MHA in PyTorch

- There are 2 ways of doing this.
    - The simple way is to just create a wrapper class that calls as many instances of the SelfAttention module as we want heads. The problem here is that this will become a sequential operation, where we are computing each head’s operation one-by-one, making the total computation time insanely high.
    - The efficient way of doing it is to split the required context vector dimensions across multiple heads. This way each head will  operate on hidden dimension of $context\_dim/num\_heads$ . The outputs for the heads are concatenated at the end to give us context vector of hidden dimensions $context\_dim$. 
    
    This method is efficient because what this means is we are just essentially splitting the weight matrices for Q,K,V across different heads by leveraging tensor reshaping and transpose operations. This is efficient because the heads’ operations are not sequential,  but rather being done in parallel using batched matrix multiplication.