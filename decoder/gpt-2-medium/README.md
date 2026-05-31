---
tags:
- task: Text Generation
- framework: PyTorch
- license: MIT
- architecture: Decoder-only
- llm
- decoder
- language-model
- gpt
---

# gpt-2-medium

**Author:** openai  
**Task:** Text Generation  
**Framework:** PyTorch  
**License:** MIT  
**Architecture:** Decoder-only

## Description

GPT-2 Medium is a 345M parameter Transformer-based language model trained on WebText. It generates coherent and fluent text across a wide range of topics.

## Model Card

# GPT-2 Medium

GPT-2 Medium (345M) by OpenAI, trained on ~40GB of internet text.

## Usage
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
```


---
*Published via [Aiva AI Forge](https://github.com/MDFJ-Free-AI-Hub)*
