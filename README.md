# 📧 Spam Detection – Data Analysis & Machine Learning

This project focuses on detecting spam emails using a combination of natural language processing (NLP), feature engineering, and machine learning techniques. Built in Jupyter Lab using Python, the analysis is based on the `spam_ham_dataset.csv` from Kaggle.

---

## 🧭 Introduction

In an age where email communication is essential, filtering spam accurately is critical. This project explores how one approaches spam detection, starting from raw text data to building a high-performing machine learning model. By combining traditional NLP with custom feature engineering, the final model not only learns from the content of emails but also leverages behavioral patterns often found in spam messages.

---

## 🚀 Project Overview

The primary goal of this project was to build an effective spam classifier. This includes data preprocessing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and visualization.

---

## 🧹 Preprocessing & Feature Engineering

- **Text cleaning**: Lowercasing, punctuation removal, stopword filtering, and stemming using `NLTK`.
- **TF-IDF Vectorization**: Represented email content numerically with `TfidfVectorizer`.
- **Custom engineered features**:
  - Number of uppercase words
  - Count of punctuation marks
  - Email/message length
  - Ratio of digits to letters
  - Number of URLs
  - Presence of reply indicators (`"re:", "fw:"`)
  - Presence of suspicious keywords (e.g., “free”, “buy”, “urgent”)

---

## 📊 Exploratory Data Analysis (EDA)

Used `Seaborn` and `Matplotlib` to visualize:
- Class distribution (spam vs. ham)
- Message length and punctuation usage across classes
- Suspicious keyword frequency
- Custom feature distribution comparisons between spam and ham

---

## 🤖 Model Training & Evaluation

- **Models used**:
  - `RandomForestClassifier`
  - `MultinomialNB` (with hyperparameter tuning using `GridSearchCV`)
- **Cross-validation**: 5-fold cross-validation to ensure stability
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score, and Confusion Matrix

---

## 🧪 Results & Insights

- The **hybrid model** combining TF-IDF and engineered features achieved **~97.97% accuracy**, outperforming the base TF-IDF-only model (97.1%).
- Engineered features such as digit-to-letter ratio, punctuation count, and presence of suspicious keywords significantly improved performance.
- A visual comparison of both models' performance was plotted to showcase the performance gain from feature engineering.

---

## 🔍 Example Predictions

Here are some example predictions made by the trained model:

| Email Snippet                                 | Predicted Label |
|----------------------------------------------|------------------|
| "Congratulations! You’ve won a $500 gift card!" | Spam             |
| "RE: Meeting notes from yesterday"            | Ham              |
| "Limited time offer – act now and save big!"  | Spam             |
| "Can we reschedule the interview for tomorrow?" | Ham              |

---

## 📁 Tools & Libraries

- `Pandas`, `NumPy` – data manipulation
- `NLTK`, `re`, `string` – text preprocessing
- `scikit-learn` – ML models, vectorization, and evaluation
- `Matplotlib`, `Seaborn` – visualization

---

## 📂 Dataset

Dataset used: [`spam_ham_dataset.csv`](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🧾 Conclusion

This project showcases how a structured approach to spam detection, combining NLP with feature engineering, can lead to a powerful and interpretable model. With high accuracy and flexibility, this system could be extended to real-time email classification systems or embedded into enterprise-level tools.

It also highlights the importance of exploratory analysis, domain knowledge, and creativity in designing features that go beyond traditional methods. A fantastic exercise in practical machine learning and real-world problem-solving.
