# 📰 Fake News Detection using Deep Learning, Machine Learning & Custom Stacking Ensemble

This project compares **Deep Learning** (Uni-LSTM, Bi-LSTM, Uni-GRU, Bi-GRU, CNN) and **Machine Learning** (SVM, Logistic Regression, Decision Tree, KNN, Random Forest) models for **Fake News Detection** using the **ISOT Fake News Dataset**.

A final **stacked ensemble model** is constructed using predictions from all models. This ensemble achieves **up to 99.85% accuracy**, outperforming individual base models.

---

## 📁 Dataset: ISOT Fake News Dataset

| File       | Description         |
|------------|---------------------|
| `True.csv` | Real news articles  |
| `Fake.csv` | Fake news articles  |

A binary `label` column is added:
- `0` → Real
- `1` → Fake

---

## 🧹 Preprocessing Pipeline

Text preprocessing steps include:
- Lowercasing
- Punctuation, number, and special character removal
- Stopword removal using **NLTK**
- Character normalization (e.g., "sooo" → "so")
- Lemmatization using **spaCy** (`en_core_web_sm`)
- Final whitespace cleanup

---

## 🔠 Tokenization & Vectorization

| Model Type | Vectorizer          | Description                                 |
|------------|---------------------|---------------------------------------------|
| DL Models  | Keras Tokenizer     | Tokenization + padding for sequence models  |
| ML Models  | TF-IDF              | For all ML models except RF2                |
| RF2 Model  | CountVectorizer     | Used only for Random Forest 2 (TF input)    |

---

## 🧠 Deep Learning Architectures

| Model     | Layers |
|-----------|--------|
| Uni-LSTM  | Embedding → LSTM → Dense + Dropout → Sigmoid |
| Bi-LSTM   | Embedding → Bidirectional LSTM → Dense → Sigmoid |
| Uni-GRU   | Embedding → GRU → Dense + Dropout → Sigmoid |
| Bi-GRU    | Embedding → Bidirectional GRU → Dense → Sigmoid |
| CNN       | Embedding → Conv1D → GlobalMaxPooling → Dense → Sigmoid |

**Common DL Settings:**
- `embedding_dim = 128`
- `max_len = 300`
- `optimizer = Adam`, `loss = BinaryCrossentropy`
- `EarlyStopping` used to avoid overfitting

---

## 🧠 Machine Learning Models (Scikit-learn)

| Model                          | Parameters                         |
|--------------------------------|------------------------------------|
| SVM                            | `C=1`                              |
| Logistic Regression            | `C=1`, `max_iter=1000`             |
| Decision Tree                  | `max_depth=5`                      |
| K-Nearest Neighbors            | `k=9`                              |
| Random Forest 1 (TF-IDF)       | `n_estimators=400`, `max_depth=40` |
| Random Forest 2 (TF only)      | `n_estimators=300`, `max_depth=40` |

---

## 🧩 Stacked Ensemble Classifier

A **Random Forest-based meta-classifier** was trained on the predictions from:
- 5 Deep Learning models (Uni-LSTM, Bi-LSTM, Uni-GRU, Bi-GRU, CNN)
- 6 Machine Learning models (SVM, LR, DT, KNN, RF1, RF2)

### 🧠 Meta-model:
```python
RandomForestClassifier(n_estimators=300, max_depth=40, random_state=42)
