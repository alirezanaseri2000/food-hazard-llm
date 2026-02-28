# Food Hazard Detection with Large Language Models 🍔🔍

**SemEval-2025 Task 9 – Comparative Study**

## 📖 Overview

This project investigates the application of Large Language Models (LLMs) to food hazard detection using the SemEval-2025 Task 9 dataset. The objective is to analyze their strengths, limitations, and trade-offs in a safety-critical NLP classification setting.

We explore and compare three distinct paradigms:

* **Prompt-Based Inference** (Zero-shot / Few-shot)
* **Supervised Fine-Tuning (SFT)**
* **Retrieval-Augmented Generation (RAG)**

## 📊 Dataset

**Source:** SemEval-2025 Task 9 – Food Hazard Detection Challenge

Each record in the dataset contains the following attributes:

* `title`
* `text`
* `hazard-category` (10 classes)
* `product-category` (22 classes)

> ⚠️ **Important:** The dataset is not included in this repository due to licensing/distribution constraints.

**Setup Instructions:**
Download the dataset from the official SemEval source and place the following files inside a `data/` directory at the root of the project:

```text
data/
├── incidents_train.csv
├── incidents_valid.csv
└── incidents_test.csv

```

## 🧠 Methods

### 1. Prompt-Based Inference

* **Technique:** Zero-shot and few-shot prompting.
* **Model:** Instruction-tuned LLM (`llama3.1:8b`).
* **Configuration:** Deterministic decoding (temperature = 0) with label-restricted outputs to strictly match the category spaces.

### 2. Supervised Fine-Tuning (SFT)

* **Architecture:** Pretrained transformer backbone.
* **Training Variants:** * Multi-task classification (jointly predicting hazard and product).
* Product-only fine-tuning.


* **Loss Function:** Cross-entropy loss.
* **Selection:** Validation-based model selection.

### 3. Retrieval-Augmented Generation (RAG)

* **Embedding Model:** `all-MiniLM-L6-v2`.
* **Vector Store:** Similarity search via FAISS (cosine similarity).
* **Experiments:** Top-$k$ retrieval experiments ($k=1, 3, 5$).
* **Setup:** Hybrid configuration combining hazard definitions with product examples in the context window.

## 📈 Evaluation

All approaches are evaluated under a consistent experimental setup. Special attention is given to model performance on rare and long-tail hazard categories.

**Metrics used:**

* **Macro-F1** *(Primary metric due to heavy class imbalance)*
* Micro-F1
* Weighted-F1
* Accuracy
* Per-class analysis

## 🚀 How to Run

### 1. Install Dependencies


*(Optional)* If running RAG, ensure FAISS is installed:

```bash
pip install faiss-cpu

```

*(Optional)* If using local LLM inference with Ollama, ensure the service is running and the model is pulled:

```bash
ollama run llama3.1:8b

```

### 2. Execute Experiments

**Prompt-Based Inference:**

```bash
python prompting/zero_shot.py

```

**Supervised Fine-Tuning:**

```bash
# For multi-task training (hazard + product)
python sft/train_multi_task.py

# For product-only fine-tuning
python sft/train_product_only.py

```

**Retrieval-Augmented Generation (RAG):**

```bash
# Base RAG pipeline
python rag/run_rag.py

# Hybrid RAG pipeline
python rag/run_hybrid.py

```

## 🔑 Key Findings

* **SFT is the strongest for hazards:** Supervised fine-tuning provides the most robust hazard classification performance overall.
* **Task separation helps SFT:** Product-only SFT significantly improves product classification compared to joint multi-task training.
* **RAG has mixed results:** RAG offers modest improvements for product classification but fails to enhance hazard detection accuracy.
* **Format $\neq$ Accuracy:** High output formatting compliance from the LLM does not inherently guarantee high classification accuracy.

📄 *For a detailed analysis, full evaluation tables, and a structured error analysis, please refer to `report.pdf`.*

## 🔬 Reproducibility Notes

To ensure reproducibility, the following standards were maintained across all experiments:

* Official dataset splits were strictly used.
* Random seeds were fixed wherever applicable.
* Generation settings were completely deterministic (e.g., `temperature=0`).
* FAISS utilized cosine similarity with normalized embeddings.
* All provided scripts and configurations match the exact setups used for the final report.

