# 🐦 EmoTweet: End-to-End Twitter Sentiment Classification

An end-to-end Natural Language Processing (NLP) pipeline designed to classify streaming and batch Twitter data into **Positive**, **Negative**, and **Neutral** sentiment categories. EmoTweet cleans raw, noisy social media text through regex sanitization, lemmatization, and sub-word TF-IDF representations before evaluating multi-class machine learning classifiers.

## 📖 Overview
Social media streams present distinct NLP challenges: informal syntax, emojis, hashtags, user mentions, abbreviations, and misspellings. 

**EmoTweet** addresses these challenges by implementing:
- Robust text sanitation pipelines specifically calibrated for unstructured tweet semantics.
- Sub-word and character/word-level TF-IDF feature extractions to mitigate out-of-vocabulary (OOV) tokens.
- Systematic benchmarking of probabilistic, linear, and margin-based classifiers to determine the optimal trade-off between inference speed and prediction performance.

## 📊 Dataset Characteristics
- **Volume:** 50,000+ labeled tweet records
- **Target Classes:** 
  - `Positive`
  - `Neutral`
  - `Negative`
- **Data Attributes:** Raw tweet text, target sentiment labels, timestamps, and tweet IDs.

## 🧠 NLP & Preprocessing Pipeline
1. **Noise Removal & Regex Sanitation:**
   - Stripped URLs (`http\S+|www\S+`), user mentions (`@username`), and hashtag symbols (`#`) while retaining the semantic root word.
   - Decoded HTML entities and eliminated punctuation and excessive whitespace.
2. **Text Normalization:**
   - Case folding (lowercasing).
   - Tokenization and elimination of non-discriminative stop-words using NLTK/spaCy.
   - WordNet lemmatization to compress morphological word variants into root lemmas.
3. **Feature Engineering:**
   - Vectorized cleaned text using **Sub-word / N-gram TF-IDF** (Term Frequency–Inverse Document Frequency), capturing contextual bi-grams and handling colloquial social media vocabulary.

## 📈 Model Benchmarking & Results
Multiple classical machine learning architectures were trained, evaluated, and tuned via cross-validation and hyperparameter optimization (GridSearchCV):

| Model Architecture | Precision | Recall | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naïve Bayes** | 0.81 | 0.80 | 0.80 | 80.4% |
| **Logistic Regression** | 0.85 | 0.84 | 0.84 | 84.8% |
| **Support Vector Machine (SVM)** | **0.88** | **0.87** | **0.87** | **87.5%** |

## 🛠 Tech Stack & Tools
- **Language:** Python 3.8+
- **NLP & Text Mining:** NLTK, spaCy, Regular Expressions (`re`)
- **Machine Learning:** scikit-learn (TF-IDF, Naïve Bayes, Logistic Regression, LinearSVC/SVC, GridSearchCV)
- **Data Manipulation:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn (Confusion matrices, ROC-AUC curves, classification reports)
