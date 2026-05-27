# Amazon Reviews Sentiment Analysis using NLP

## Overview

This project implements a complete **Natural Language Processing (NLP) pipeline** to perform **sentiment analysis on Amazon product reviews**. The goal is to classify customer feedback as **positive or negative** using classical machine learning techniques.

The workflow covers:

* Exploratory Data Analysis (EDA)
* Text preprocessing and cleaning
* Feature extraction using Bag-of-Words
* Model training with multiple classifiers
* Performance evaluation using standard classification metrics

---

## Dataset Description

The dataset consists of Amazon product reviews with structured fields, including:

* `verified_reviews`: Textual review content
* `rating`: Numerical rating (typically 1–5)
* `feedback`: Binary sentiment label

  * `1` → Positive
  * `0` → Negative

### Key Observations:

* Strong class imbalance:

  * ~2800 positive reviews
  * ~250 negative reviews
* Reviews vary significantly in length
* Missing values exist in the `verified_reviews` column (handled during preprocessing)

---

## Exploratory Data Analysis (EDA)

Several exploratory steps were performed to understand the dataset:

### 1. Data Inspection

* `.info()` used to check data types and missing values
* `.describe()` used for statistical summaries

### 2. Rating Distribution

* Count plots reveal skew toward higher ratings

### 3. Review Length Analysis

* A new feature `length` was created
* Histogram shows distribution of review lengths

### 4. Sentiment Distribution

* Significant imbalance between positive and negative classes

### 5. WordCloud Visualization

* Positive and negative reviews were visualized separately
* Helps identify commonly used words in each sentiment category

---

## Text Preprocessing Pipeline

A custom preprocessing function (`message_cleaning`) was implemented with the following steps:

### Steps:

1. **Remove punctuation**
2. **Tokenization**
3. **Stopword removal** using NLTK

### Implementation Details:

* Uses Python’s `string.punctuation`
* Uses NLTK English stopwords
* Outputs a cleaned list of tokens for each review

Example transformation:

* Raw: `"This product is AMAZING!!!"`
* Cleaned: `["product", "amazing"]`

---

## Feature Engineering

### Bag-of-Words Representation

The project uses **CountVectorizer** from Scikit-learn:

* Custom analyzer: `message_cleaning`
* Converts text into a sparse matrix of token counts
* Vocabulary extracted from the entire dataset

### Result:

* High-dimensional sparse feature matrix
* Shape: `(num_samples, num_unique_words)`

---

## Model Pipeline

### Data Splitting

* Train-test split: **80% training / 20% testing**

---

## Models Implemented

### 1. Multinomial Naive Bayes

* Suitable for discrete word count features
* Assumes feature independence
* Fast and efficient baseline model

### 2. Logistic Regression

* Linear model for binary classification
* Learns weights for each feature
* Handles high-dimensional sparse data well

### 3. Gradient Boosting Classifier

* Ensemble method using decision trees
* Sequential learning to reduce errors
* More complex, captures non-linear relationships

---

## Evaluation Metrics

Each model is evaluated using:

* **Confusion Matrix**
* **Precision**
* **Recall**
* **F1-score**
* **Support**

Visualization:

* Heatmaps used for confusion matrices via Seaborn

---

## Results and Performance

### 1. Multinomial Naive Bayes

* Performs well on text classification tasks
* High accuracy on majority class (positive reviews)
* Slight bias due to class imbalance

**Key Insight:**

* Strong baseline model with fast training time

---

### 2. Logistic Regression

* Improved balance between precision and recall
* Better generalization compared to Naive Bayes

**Key Insight:**

* Handles feature interactions better than Naive Bayes

---

### 3. Gradient Boosting Classifier

* More flexible model capturing non-linearities
* Higher computational cost
* Risk of overfitting on imbalanced data

**Key Insight:**

* Performance depends heavily on hyperparameters and dataset balance

---

## Comparative Analysis

| Model               | Strengths                     | Weaknesses                     |
| ------------------- | ----------------------------- | ------------------------------ |
| Naive Bayes         | Fast, simple, strong baseline | Assumes independence           |
| Logistic Regression | Balanced, interpretable       | Linear decision boundary       |
| Gradient Boosting   | Powerful, flexible            | Slower, sensitive to imbalance |

---

## Key Insights

* **Class imbalance significantly affects performance**, especially recall for negative reviews
* **Bag-of-Words is effective but limited** (no semantic understanding)
* **Stopword removal improves signal-to-noise ratio**
* Logistic Regression provides the best trade-off between simplicity and performance

---

## Project Workflow Summary

1. Data Loading
2. Data Cleaning (handling nulls)
3. Exploratory Data Analysis
4. Feature Engineering (text → numerical)
5. Train-Test Split
6. Model Training (3 algorithms)
7. Evaluation using classification metrics
8. Visualization of results

---

## Technologies Used

* Python
* Pandas & NumPy
* Matplotlib & Seaborn
* NLTK
* Scikit-learn
* WordCloud

---

## Conclusion

This project demonstrates a complete **end-to-end NLP classification pipeline** for sentiment analysis using traditional machine learning methods. It highlights the importance of preprocessing, feature extraction, and model selection when working with textual data.

The implementation provides a strong foundation for more advanced NLP techniques such as:

* TF-IDF vectorization
* Word embeddings (Word2Vec, GloVe)
* Deep learning models (LSTM, Transformers)

---
