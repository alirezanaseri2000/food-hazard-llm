Food Hazard Detection with Large Language Models

SemEval-2025 Task 9 – Comparative Study

Overview

This project investigates the application of Large Language Models (LLMs) to food hazard detection using the SemEval-2025 Task 9 dataset.

Three paradigms are explored and compared:

Prompt-Based Inference (Zero-shot / Few-shot)

Supervised Fine-Tuning (SFT)

Retrieval-Augmented Generation (RAG)

The objective is to analyze their strengths, limitations, and trade-offs in a safety-critical NLP classification setting.

Dataset

Dataset: SemEval-2025 Task 9 – Food Hazard Detection Challenge

Each record contains:

title

text

hazard-category

product-category

Label spaces:

Hazard categories: 10 classes

Product categories: 22 classes

⚠️ The dataset is not included in this repository.
Download it from the official SemEval source and place the following files inside a data/ directory:

data/
├── incidents_train.csv
├── incidents_valid.csv
└── incidents_test.csv

Methods
1. Prompt-Based Inference

Zero-shot and few-shot prompting

Instruction-tuned LLM (llama3.1:8b)

Deterministic decoding (temperature = 0)

Label-restricted outputs

2. Supervised Fine-Tuning (SFT)

Pretrained transformer backbone

Multi-task classification (hazard + product)

Product-only fine-tuning

Cross-entropy loss

Validation-based model selection

3. Retrieval-Augmented Generation (RAG)

Embedding model: all-MiniLM-L6-v2

Similarity search via FAISS (cosine similarity)

Top-k retrieval experiments (k=1,3,5)

Hybrid configuration (hazard definitions + product examples)

Evaluation

All approaches are evaluated under a consistent experimental setup using:

Accuracy

Macro-F1 (primary metric due to class imbalance)

Micro-F1

Weighted-F1

Per-class analysis

Special attention is given to rare and long-tail hazard categories.

How to Run
Install dependencies

pip install -r requirements.txt

If running RAG:

pip install faiss-cpu

If using local LLM inference with Ollama:

ollama run llama3.1:8b

Prompt-Based Inference

python prompting/zero_shot.py

Supervised Fine-Tuning

python sft/train_multi_task.py
python sft/train_product_only.py

RAG

python rag/run_rag.py
python rag/run_hybrid.py

Key Findings

Supervised fine-tuning provides the most robust hazard classification performance.

Product-only SFT significantly improves product classification compared to multi-task training.

RAG offers modest improvements for product classification but does not enhance hazard detection.

High output formatting compliance does not guarantee high classification accuracy.

See report.pdf for detailed analysis, evaluation tables, and structured error analysis.

Reproducibility Notes

Official dataset splits were used.

Random seeds fixed where applicable.

Deterministic generation settings (temperature=0).

FAISS cosine similarity with normalized embeddings.

All experiments are designed to be reproducible with the provided scripts and configuration.
