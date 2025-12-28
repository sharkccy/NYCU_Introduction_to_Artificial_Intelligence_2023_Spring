# HW2: Sentiment Analysis and Language Modeling

Author: Chung-Yu Chang (張仲瑜)  
Student ID: 110612117

## Introduction
This project explores multiple NLP techniques for sentiment analysis on the IMDB dataset, from classical N-gram models to RNN/LSTM architectures and BERT fine-tuning.

## Methodology

### Part 0: Data Preprocessing
- Stopword removal using NLTK to drop non-informative tokens (e.g., "on", "at", "and").
- Custom cleaning via `preprocessing_function`: strip HTML tags (<br />), remove numbers, lowercase text, and drop special characters or duplicate spaces.

### Part 1: N-gram Modeling
- Bi-gram language model to predict word sequences.
- Evaluation with Perplexity (lower = more confident/accurate distribution fit).

### Part 2 & 3: Deep Learning Models
- RNN / LSTM: LSTM-based architecture with `nn.Embedding` for token vectors and dropout to reduce overfitting.
- BERT fine-tuning: `distilbert-base-uncased` with a custom MLP head (Linear -> Dropout -> Linear) for sentiment classification.

## Implementation Details
- main.py: entrypoint for training/evaluation via CLI args.
- ngram.py: n-gram co-occurrence and probability logic.
- rnn.py: LSTM model definition and data processing.
- bert.py: DistilBERT fine-tuning wrapper.
- preprocess.py: text cleaning and tokenization utilities.

## Results
- BERT performance: F1 = 0.9209, Precision = 0.9213, Recall = 0.9209 after the first epoch.
- Observation: preprocessing (especially stopword removal) significantly stabilizes and improves deep learning model performance.