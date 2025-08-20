# COVID-19 NLP Text Classification

## Overview:
This project implements a text classification model to predict the sentiment of COVID-19 related tweets. The dataset is sourced from Kaggle: COVID-19 NLP Text Classification.
The model uses bert-tiny from Hugging Face Transformers for efficient training on small datasets, with GPU acceleration for fast experimentation.

## Features:
- Sentiment prediction for COVID-19 tweets.
- Tokenization and encoding with AutoTokenizer.
- Training and validation with GPU support.
- Metrics tracked: Accuracy, Precision, Recall, F1-score.
- Generates a CSV with predictions for test data.

## Installation:
Clone the repository and install required packages:

```bash
git clone https://github.com/zaralrubaie/NLP-text-classification-problem-solve.git
cd NLP-text-classification-problem-solve
pip install torch transformers scikit-learn pandas tqdm
```
## Usage
1. Make sure you have installed the required packages (see Installation section).
2. Open the `coronavirus_tweets_nlp.ipynb` notebook in Jupyter or Colab.
3. Run the cells in order:
   - Load and preprocess data
   - Tokenize tweets
   - Create datasets and DataLoaders
   - Initialize model and optimizer
   - Train the model
   - Evaluate and save test predictions
     
4. Check the generated `test_predictions.csv` for model outputs.
   
## Metrics:
- Test Accuracy: 0.6685
- Test Precision: 0.6771
- Test Recall: 0.6685
- Test F1-score: 0.6700

## Note
manual training was used because trainer can not be function on kaggle since it has old version of transformers 

## Trainer vs Manual Loop:
- Manual loop used here for speed and simplicity on a small dataset.
- Trainer API automates training but adds overhead; better for larger datasets.
- GPU acceleration is recommended for faster training.
  
## For better performance
- bert-base-uncased or distilbert-base-uncased
- Slightly larger max_length (150–200) for tokenization
- More training epochs or learning rate scheduling

