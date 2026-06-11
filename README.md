# Sentiment Analysis on Movie Reviews

## Project Overview

This project implements an end-to-end Sentiment Analysis system using Natural Language Processing (NLP) and Machine Learning techniques. The objective is to classify movie reviews as Positive or Negative based on their textual content.

## Dataset

* IMDb Movie Reviews Dataset
* 50,000 movie reviews
* Binary sentiment labels: Positive and Negative

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-Learn
* VADER
* Hugging Face Transformers
* DistilBERT
* Streamlit

## Project Workflow

1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis (EDA)
3. TF-IDF Feature Extraction
4. Logistic Regression
5. Support Vector Machine (SVM)
6. VADER Sentiment Analysis
7. DistilBERT Fine-Tuning
8. Error Analysis
9. Streamlit Deployment

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| VADER               | 69%      |
| Logistic Regression | 88%      |
| SVM                 | 88%      |
| DistilBERT          | 90%      |

DistilBERT achieved the highest performance due to its ability to capture contextual information within text.

## Streamlit Application

The trained model was deployed using Streamlit, allowing users to enter movie reviews and receive real-time sentiment predictions.

## Future Improvements

* Multi-class sentiment classification
* Sarcasm detection
* Aspect-based sentiment analysis
  
## Author

Kavya
Aspiring Data Scientist.
