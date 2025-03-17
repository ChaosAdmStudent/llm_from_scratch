# Different categories of fine-tuning 

1. **Instruction fine-tuning**: Fine tuning the model using instructions included in the prompt to improve its ability to understand and execute tasks described in natural language prompts. It is a generalized model that can do many different things. Needs much larger dataset and compute power to be proficient.

2. **Classification fine-tuning**: Fine-tuning the model to classify text into one of many categories. Example: spam vs not spam. Can only output one of the classes. Can do with smaller dataset and less compute. 

# Classification Fine Tuning 

1. Replace output layer with classification head. 
2. Freeze earlier layers of pretrained model because lower layers usually capture most general language structure. 
3. Layers close to classification head learn more task-specific stuff, so unfreezing here helps. 
4. In LLMs, since our input was `(B,N,num_tokens)` and we get output shape `(B,N,out_classes)`, when we do classification fine-tuning for something like spam or not-spam, we will get a label for each token but we want 1 label for the whole input prompt instead. For this, we look at the label values for the last token. This is because of causal attention coding. The last token has processed multi-attention after looking at every other token before it, so it will have the most valuable context when looking at any operations that needs to be done for the entire prompt. 