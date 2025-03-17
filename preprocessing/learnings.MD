These are all the notes and learnings from Chapter 2 (Preparing input data for LLMs)

- Word2Vec is a famous word embedding generation model. It generates word embeddings by learning to predict the context of a word given the target word, or vice versa. The main idea is that words that appear in similar contexts tend to have similar meanings. Thus, if we look at 2D projection of word embeddings coming out of Word2Vec, we can see similar words are grouped together in the graph.
- LLMs often make their own embedding layers in the beginning. Optimizing embeddings in LLM architecture itself helps to fine tune according to the data and task at hand.
- General work flow for preparing data:
    - Text → Tokens → Token IDs → Embedding Vector
    - Token to Token IDs is just about creating a dictionary that maps alphabetically sorted words to unique numbers.

- SPECIAL CONTEXT TOKENS are required to handle unknown or unseen tokens in new text so that our tokenizer doesn’t give errors and still knows how to handle those. Special tokens also help the model understand specific context that helps it to learn more effectively. Common special context tokens in LLMs are:
    - [BOS]  → Beginning of Sequence
    - [EOS]  → End of Sequence
    - [PAD]   → Padding.
        - This used to be in use since models train in batches and to maintain a fixed token length for all training examples, this token would be used repeatedly to maintain equal token length for all training examples in a batch. However, this is not used anymore because we use a mask mechanism in the attention such that padded tokens are not attended to, so these padding tokens don’t have any effect.
    - [|unk|] → Unknown words
        - This is also not used these days because GPT-like LLMs use Byte-pair encoding, which breaks down words into subwords as tokens.

- Byte Pair Encoding creates a  vocabulary that iteratively merges characters into subwords and subwords into words based on some frequency cutoff. It helps avoid problems of facing unknown words by iteratively looking at a given word, which in the worst case scenario would be a collection of single characters (a to z).
    - In the first iteration, the vocabulary is just the single english alphabet characters a to z.
    - In subsequent iterations over different words, it sees characters that are occurring together frequently. For instance, “d” and “e” in english are commonly occurring; example: define, depend, hidden etc.
    - Thus, “d” and “e” are merged to form a new token “de”
    - The merging decision is based on some frequency cutoff value.

- Next, we make input-target pairs for the LLM to process. We do this through a sliding window approach.
- The tokens are then converted into embedding vectors. The dimensionality of the embedding vector is a hyperparameter. The larger, more the context about the word and the model has more chance to capture nuanced relationships about the word, but the computation also goes up. It’s a tradeoff.
- The embedding layer in PyTorch helps us to do this. It can be mathematically proved to be a more efficient implementation of one-hot encoding of token IDs multiplied with a Linear layer. Hence, we can look at the Embedding layer as a neural network layer that can be optimized using backpropagation. By using the Embedding layer instead of one-hot encoding followed by matrix multiplication with Linear layer weights is that we skip over a lot of multiplication with 0s and save computation.
- Self-attention mechanism is position-agnostic and without positional information, LLM has no way of knowing the order/position of words which is very important for languages. Thus, we need to add positional embeddings to the token embeddings to get input embedding. 

$Input Embedding = Positional Embeddings + TokenEmbeddings$
- There are 2 types of positional embeddings:
    - Absolute positional embedding: Here, the embeddings are added based on fixed positions. This requires that the length of text is not longer than the length used during training.
    - Relative Positional Embeddings: Here, the embeddings are added based on relative distance/position between tokens. The focus here is to make model learn “how far apart are tokens” instead of “at which position is token”.

- Original transformer used fixed, predetermined absolute positional embeddings (Sinusoidal Positional Embeddings) to add context about the positions in the input tokens. However, ChatGPT today uses absolute PE that are optimized during training.