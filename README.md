# Build GPT-2 Final Project

## Overview
This project implements a GPT-2–style Transformer model to perform multiple natural language processing tasks, including sentiment classification, paraphrase detection, and text generation. The goal is to evaluate how effectively pretrained language representations transfer to downstream tasks while minimizing the number of parameters updated during training.

---

## Problem
Large language models achieve strong performance across many NLP tasks, but full fine-tuning can be computationally expensive. This project investigates:
- how well GPT-2 representations transfer across tasks  
- whether lightweight fine-tuning is sufficient  
- how dataset characteristics affect downstream performance  

---

## Approach

### Model Architecture
- GPT-2–style Transformer decoder implemented in PyTorch  
- Components:
  - masked multi-head self-attention  
  - position-wise feed-forward layers  
  - residual connections  
  - layer normalization  
- Tokenization: Byte Pair Encoding (BPE)

### Task Adaptation
The model is adapted to different tasks using lightweight prediction heads:

- **Sentiment Classification**
  - Input: single sentence  
  - Output: binary sentiment label  

- **Paraphrase Detection**
  - Input: sentence pairs  
  - Output: paraphrase / non-paraphrase  

- **Text Generation**
  - Input: partial sonnet (first 3 lines)  
  - Output: generated continuation (autoregressive)

---

## Training Strategies

We compare two fine-tuning approaches:

1. **Last-Layer Fine-Tuning**
   - Freeze GPT-2 parameters  
   - Train only the final classification layer  
   - Efficient but limited adaptability  

2. **Full-Model Fine-Tuning**
   - Update all model parameters  
   - Higher computational cost  
   - Better performance on complex tasks  

---

## Datasets

- **SST (Stanford Sentiment Treebank)**  
  Short movie review sentences with sentiment labels  

- **CFIMDB**  
  Longer movie reviews with binary sentiment labels  

- **Quora Question Pairs**  
  Sentence pairs labeled as paraphrase or non-paraphrase  

- **Shakespeare Sonnets**  
  Corpus used for conditional text generation  

---

## Results

| Task                     | Dataset | Metric                  | Result        |
|--------------------------|--------|--------------------------|--------------|
| Sentiment Classification | SST    | Accuracy                 | 0.452        |
| Sentiment Classification | CFIMDB | Accuracy                 | 0.780        |
| Paraphrase Detection     | Quora  | Accuracy / F1            | **0.813 / 0.795** |
| Text Generation          | Sonnets| Training Loss            | 5.15 → 3.87  |

### Key Findings
- GPT-2 representations transfer well to semantic similarity tasks  
- Best performance achieved on paraphrase detection  
- Longer text inputs (CFIMDB) yield better sentiment performance than short sentences (SST)  
- Generated text captures stylistic patterns but may lose coherence over longer sequences  

---

## Tech Stack
- PyTorch  
- HuggingFace Transformers (tokenizer)  
- NumPy  
- tqdm  
- sacrebleu (for evaluation)

---

## Repository Structure

```bash
models/              # GPT-2 model implementation  
datasets/            # Dataset loading and preprocessing  
training/            # Training scripts  
evaluation/          # Evaluation metrics and scripts  
utils/               # Helper functions  
