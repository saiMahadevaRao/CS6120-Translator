---

# Document Summarization and Translation System

Welcome to the Document Summarization and Translation project! This project bridges language barriers by summarizing and translating multilingual documents. The system is designed for ease of use and performance, offering translation and summarization for low-resource languages using custom-trained models.

---

## Project Overview

This repository implements a **summarization** and **translation** system. It provides a simple pipeline that generates concise summaries from documents and translates them into various languages. 

### Key Features:
- Custom-trained Transformer-based translation model.
- Algorithmic-based document summarization without pre-trained models.
- Evaluation of the generated summaries using ROUGE scores.

---

## Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/document-summarization-translation.git
cd document-summarization-translation
```

### 2. Install Required Packages

Make sure you have Python 3.7+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

The project uses the following key libraries:
- `transformers`: For implementing the Transformer-based translation models.
- `datasets`: For accessing and managing datasets like WikiLingua.
- `rouge_score`: To evaluate the summarization performance.
- `torch`: For running the models on GPU.

To ensure GPU support for faster computation, install the right version of PyTorch that supports your machine’s CUDA version (if applicable). You can check available versions [here](https://pytorch.org/get-started/locally/).

---

## How to Run the Project

### 1. Load and Preprocess the Data

The project uses the **WikiLingua** dataset for translation and summarization tasks. Make sure to load the data from Hugging Face:

```python
from datasets import load_dataset

# Example to load the English part of the dataset
dataset = load_dataset("esdurmus/wiki_lingua", "english")
```

You can preprocess the data according to your requirements (e.g., lowercasing, removing numbers, etc.).

### 2. Summarization and Translation

The main pipeline provides a simple algorithm for summarization and translation. Here’s an example of how to summarize and translate:

```python
# Summarize a document (max 100 characters per summary)
summary = summarize_document(document, max_len=100)

# Translate the summary into a target language
translation = translate_summary(summary, target_language="es")
```

You can modify the code within the provided Jupyter Notebook for experimenting with different models and approaches.

### 3. Evaluate the Model

You can evaluate the quality of the summaries using the ROUGE score:

```python
# Compute ROUGE score
rouge_scores = compute_rouge_score(reference_summaries, generated_summaries)
```

---

## Notebook Walkthrough

For detailed exploration, the core implementation resides in the `Translator.ipynb` Jupyter Notebook. To run the notebook:

1. Install Jupyter:
   ```bash
   pip install jupyter
   ```
2. Launch the notebook:
   ```bash
   jupyter notebook Translator.ipynb
   ```

In the notebook, you will find code for:
- Loading datasets.
- Preprocessing.
- Summarization.
- Translation using a custom Transformer-based model.
- Evaluating generated summaries using ROUGE.

---

## Project Structure

- `Translator.ipynb`: Main notebook for summarization and translation.
- `requirements.txt`: All the dependencies required for the project.
- `README.md`: This file.
- `saved_models/`: Trained models stored at different epochs.

---

## Contributing

If you would like to contribute to this project, feel free to open a pull request or submit an issue. Let’s make document translation more accessible for everyone!

---

