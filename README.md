This project compares Deep Learning (LSTM, GRU, CNN) and Machine Learning (SVM, Logistic Regression, Decision Tree, KNN, Random Forest) approaches for **Fake News Detection** using the **ISOT Fake News Dataset**. It achieves **>99% accuracy** with several models and provides detailed comparisons.

---

## 📦 Dataset: ISOT Fake News Dataset

| File       | Description         |
|------------|---------------------|
| `True.csv` | Real news articles  |
| `Fake.csv` | Fake news articles  |

A `label` column was added:
- `0` → Real
- `1` → Fake

---

## 🧹 Preprocessing Overview

### ✅ Shared Preprocessing (DL + ML)
- Combined and used only the `text` column.
- Converted to lowercase.
- Removed punctuation, numbers, and special characters.
- Removed stopwords using **NLTK**.
- Normalized repeating characters (e.g., "cooool" → "cool").
- Used **spaCy** for lemmatization (`en_core_web_sm`).
- Applied whitespace cleanup.

---

## 🔠 Tokenization

| Model Type | Vectorization                     | Description                  |
|------------|-----------------------------------|------------------------------|
| DL Models  | Keras Tokenizer + Padding         | For LSTM, GRU, CNN           |
| ML Models  | TF-IDF (All except RF2)           | For SVM, LR, DT, KNN, RF1    |
| ML Models  | Count Vectorizer (TF only for RF2)| For Random Forest 2 only     |

---

## 🔮 Model Architectures

### 🔷 Deep Learning Models (TensorFlow/Keras)

| Model | Layers Used |
|-------|-------------|
| **LSTM** | Embedding → BiLSTM → Dense + Dropout → Sigmoid |
| **GRU**  | Embedding → BiGRU → Dense + Dropout → Sigmoid  |
| **CNN**  | Embedding → Conv1D → GlobalMaxPooling → Dense + Dropout → Sigmoid |

All DL models used:
- `embedding_dim = 128`
- `max_len = 300`
- `EarlyStopping` callback
- `BinaryCrossentropy` loss
- `Adam` optimizer

### 🔷 Machine Learning Models (Scikit-learn)

| Model                          | Parameters                     |
|--------------------------------|--------------------------------|
| SVM                            | C=1                            |
| Logistic Regression            | C=1, max_iter=1000             |
| Decision Tree                  | max_depth=5                    |
| K-Nearest Neighbors            | k=9                            |
| Random Forest 1                | n_estimators=400, depth=40     |
| Random Forest 2 (TF)           | n_estimators=300, depth=40     |

---

## 📊 Final Combined Model Comparison

| Model                          | Test Accuracy | Precision | Recall | F1-Score |
|--------------------------------|---------------|-----------|--------|----------|
| **LSTM**                       | 0.9960        | 0.9932    | 0.9983 | 0.9958   |
| **GRU**                        | 0.9950        | 0.9956    | 0.9938 | 0.9947   |
| **CNN**                        | 0.9970        | 0.9964    | 0.9974 | 0.9969   |
| SVM (C=1)                      | 0.9938        | 0.9930    | 0.9940 | 0.9935   |
| Logistic Regression (C=1)      | 0.9842        | 0.9823    | 0.9843 | 0.9833   |
| Decision Tree (max_depth=5)    | 0.9940        | 0.9901    | 0.9974 | 0.9937   |
| KNN (k=9)                      | 0.8618        | 0.8771    | 0.8241 | 0.8497   |
| Random Forest 1 (n=400, d=40)  | 0.9973        | 0.9975    | 0.9968 | 0.9972   |
| Random Forest 2 (n=300, d=40)  | 0.9976        | 0.9974    | 0.9975 | 0.9975   |


## 🚀 How to Run

1. Upload the `True.csv` and `Fake.csv` files to your Colab session.
2. Run preprocessing cells.
3. Train deep learning and machine learning models.
4. Evaluate and visualize results.

---

## 🛠 Libraries Used

- `TensorFlow`, `Keras`, `Scikit-learn`
- `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- `NLTK`, `spaCy`

---
