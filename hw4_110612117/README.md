# HW4: Attention Mechanism Analysis and NLP Exploration

Author: Chung-Yu Chang (張仲瑜)  
Student ID: 110612117

## Introduction
This project examines Transformer attention using the exBERT visualization tool on a pre-trained `distilbert-base-uncased` model, focusing on how layers and heads capture linguistic structure and sentiment.

## Methodology

### Part 1: Attention Mechanism Analysis (exBERT)
- Analyzed the sentence: "Breathing fresh air in the forest is refreshing." to inspect token interactions.
- Layer observations:
	- Layer 2: diverse head influences with varied attention distribution across the sentence.
	- Layer 6: focuses on end-of-sentence punctuation/special tokens (e.g., [SEP]), highlighting structural/positional encoding in deeper layers.
- Head contribution: inspected how one word influences predictions of others (left-side selection) and how other words affect a target word (right-side selection).

### Part 2: Sentiment and Language Modeling
- Sentiment intensity comparison for words like "Boring" vs. "Fantastic".
- Results: higher intensity for "Boring" (8.707) than "Fantastic" (3.607) in the tested context.

## Implementation Details
- Model: `distilbert-base-uncased` (Hugging Face Transformers).
- Tools:
	- exBERT for interactive attention visualization/debugging.
	- Jupyter/Colab for running analysis and classification notebooks.

## Findings & Conclusion
- Attention diversity: layers specialize from semantic relationships to boundary detection.
- Linguistic interpretation: visualization (exBERT) helps demystify how deep models prioritize tokens during inference.