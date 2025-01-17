# Multiclass Classification of Text Reviews with NLP

This project focuses on building a machine learning pipeline for multiclass classification of text reviews. The objective is to classify text reviews into three categories: Positive, Neutral, and Negative. Various NLP techniques were applied, and multiple machine learning models were trained and evaluated.

## Table of Contents
1. [Overview](#overview)
2. [Technologies Used](#technologies-used)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results](#results)
7. [Usage](#usage)
8. [Acknowledgments](#acknowledgments)

---

## Overview
Text reviews often hold valuable insights that can be extracted and classified into sentiment categories. This project utilizes Natural Language Processing (NLP) to:

- Preprocess raw text data.
- Convert text into numerical features using TF-IDF.
- Address class imbalance with techniques such as class weighting.
- Train and evaluate machine learning models.

## Technologies Used
- Python 3.12
- Libraries:
  - [spaCy](https://spacy.io/): For lemmatization and tokenization.
  - [Scikit-learn](https://scikit-learn.org/): For feature extraction and machine learning.
  - [LightGBM](https://lightgbm.readthedocs.io/): Gradient boosting framework.
  - [XGBoost](https://xgboost.readthedocs.io/): Extreme gradient boosting.
  - [CatBoost](https://catboost.ai/): Categorical boosting.
  - [Matplotlib](https://matplotlib.org/): For visualizations.
  - [Joblib](https://joblib.readthedocs.io/): Model persistence.
  - [Imbalanced-learn (SMOTE)](https://imbalanced-learn.org/): For oversampling.

---

## Data Preprocessing
1. **Lemmatization and Tokenization**:
   - Using spaCy, text data was lemmatized to normalize words and remove stopwords.

2. **TF-IDF Vectorization**:
   - Text data was converted into numerical features using the TF-IDF vectorizer with a maximum of 5000 features and n-gram range of (1, 2).

3. **Class Imbalance Handling**:
   - Initial experiments with SMOTE for oversampling were conducted but found ineffective for high-dimensional sparse data like TF-IDF.
   - Class weights were used instead for imbalance handling in models.

4. **Train-Validation-Test Split**:
   - Data was split into training, validation, and test sets using `stratify` to maintain class distribution.

---

## Model Training
The following models were trained and evaluated:
1. Logistic Regression (with class weights)
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost

---

## Evaluation Metrics
The models were evaluated using:
- **Accuracy**: Overall correctness of predictions.
- **F1-Score (Macro)**: Balances precision and recall across all classes.
- **Precision**: Fraction of relevant instances among retrieved instances.
- **Recall**: Fraction of relevant instances that were retrieved.

---

## Results
| Model            | Accuracy | F1-Score | Precision | Recall |
|------------------|----------|----------|-----------|--------|
| Logistic Regression | 0.77     | 0.60     | 0.59      | 0.65   |
| Random Forest      | 0.81     | 0.58     | 0.61      | 0.57   |
| XGBoost            | 0.84     | 0.58     | 0.67      | 0.55   |
| LightGBM           | 0.84     | 0.59     | 0.68      | 0.56   |
| CatBoost           | 0.76     | 0.60     | 0.58      | 0.66   |

LightGBM and XGBoost achieved the best performance across all metrics.

---

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sada-consultancy/pet_project.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess Data**:
   Place your raw data in the `./files/input/` directory and run:
   ```bash
   python preprocess.py
   ```

4. **Train Models**:
   ```bash
   python train.py
   ```

5. **Evaluate Models**:
   ```bash
   python evaluate.py
   ```

---

## Acknowledgments
Special thanks to the developers of the tools and libraries used in this project, as well as the open-source community for their contributions to NLP and machine learning.

