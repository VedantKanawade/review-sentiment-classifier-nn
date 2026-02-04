# Sentiment Analysis: TF-IDF & SBERT Embeddings with Neural Networks

Predicting sentiment (positive/negative) from customer reviews using advanced NLP techniques. This project demonstrates an end-to-end applied machine learning workflow including data cleaning, feature engineering, embedding generation, neural network model training, hyperparameter tuning, evaluation, and optional interactive application deployment.A detailed Project report is included in the reports folder
---

## Table of Contents
- Problem Statement
- Dataset
- Data Cleaning & Preprocessing
- Feature Engineering & Embeddings
- Exploratory Data Analysis
- Models
- Hyperparameter Tuning
- Evaluation & Metrics
- Optional Interactive Demo
- Business / Project Insights
- Future Work


---

## Problem Statement
Understanding customer sentiment is critical for product improvement, marketing, and customer retention strategies. Accurately classifying reviews as positive or negative allows businesses to make data-driven decisions and prioritize interventions.

Goal: Build robust neural network classifiers using TF-IDF and SBERT embeddings, evaluate their performance, and provide a clear, reproducible workflow.

---

## Dataset
- ~7,400 labeled customer reviews  
- Columns include:
  - `reviews` → text of the customer review  
  - `sentiments` → target label: 1 = Positive, 0 = Negative  

- Cleaned and preprocessed to remove noise, punctuation, stopwords, and standardize text formatting.

---

## Data Cleaning & Preprocessing
- Removed null or empty reviews  
- Standardized text to lowercase  
- Removed punctuation, special characters, and extra spaces  
- Optional tokenization and stopword removal for TF-IDF processing  

The cleaning process ensures the text is suitable for both sparse (TF-IDF) and dense (SBERT) embeddings.

---

## Feature Engineering & Embeddings
- **TF-IDF Embeddings**: Sparse vector representation capturing word importance  
- **SBERT Embeddings**: Dense contextual representation capturing semantic meaning  
- Embeddings are saved in `.npy` format for train, validation, and test sets  
- Data split: 70% train, 15% validation, 15% test  

---

## Exploratory Data Analysis
- Class distribution:
  - Positive: ~3,700  
  - Negative: ~3,700  
- Basic statistics on review length, word frequency, and sentiment distribution  
- Optional visualization: word clouds or embedding 2D projection (t-SNE / UMAP)

---

## Models
| Model | Description |
|-------|------------|
| Neural Network (TF-IDF) | Feedforward network with 1 hidden layer [128] neurons |
| Neural Network (SBERT) | Feedforward network with 2 hidden layers [256, 128] neurons |

Evaluation Metrics:
- Accuracy  
- F1-Score  
- Precision / Recall  
- Confusion matrix heatmap  

---

## Hyperparameter Tuning
- Hidden units: `[128], [256], [128,64], [256,128], [512,256]`  
- Dropout: `[0.1, 0.3, 0.5, 0.6]`  
- Learning rate: `[0.1, 0.001, 0.0005, 0.0001]`  
- Batch size: `[16, 32, 64, 128]`  
- Optimizer: `Adam` or `RMSprop`  

Hyperparameter tuning was performed **one parameter at a time** while keeping others constant. Validation accuracy and loss plots were generated for each configuration.

---

## Evaluation & Metrics
### Final Best Models
**TF-IDF Neural Network**
- Hidden Units: [128]  
- Dropout: 0.3  
- Learning Rate: 0.0001  
- Batch Size: 32  
- Optimizer: Adam  

**SBERT Neural Network**
- Hidden Units: [256,128]  
- Dropout: 0.1  
- Learning Rate: 0.0001  
- Batch Size: 32  
- Optimizer: Adam  

### Test Set Performance
- Confusion matrix plotted as a heatmap  
- Accuracy and F1-score reported for both models  

Key Observations:
- SBERT embeddings offered slightly better performance than TF-IDF  
- TF-IDF provides a computationally efficient alternative with competitive results  

---

## Optional Interactive Demo
- Streamlit / Dash / Gradio can be used to create a dashboard  
- Features:
  - Input review text to get predicted sentiment  
  - Switch between TF-IDF and SBERT models  
  - Display probability/confidence of prediction  

---

## Business / Project Insights
- Both TF-IDF and SBERT are effective for binary sentiment classification on moderate datasets  
- SBERT’s contextual embeddings can capture subtle semantic differences, useful for more complex or larger datasets  
- TF-IDF is sufficient for quick, computationally efficient baseline deployment  

---

## Future Work
- Extend to multi-class sentiment analysis  
- Use larger datasets to exploit SBERT advantages  
- Deploy an interactive dashboard for real-time prediction  
- Integrate explainability (e.g., SHAP) to highlight words influencing sentiment  

---
