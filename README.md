# LLM From Scratch 

In this repository, I am uploading every little code and experiment that I do to build my very own LLM from Scratch, following Sebastian Raschka's book. This also includes implementations and concepts that may not be covered in the book but I cover purely out of my own fascination. 


## List of topics that I augmented to Sebastian's book in this repo 

### Position Embeddings 

1. Sinusoidal Positional Embedding

### Attention Mechanisms 

1. Multi-Query Attention 
2. Grouped Query Attention
3. KV-Cache Mechanism for faster inference. 

## Topics to implement: 

1. Relative Position embedding
2. Flash attention

## Interesting observations 

### KV Cache inference speed up with increasing sequence lengths 
![Inference speed comparison with and without KV Cache mechanism](pretraining/plots/compare_time_kv_nokv.png)