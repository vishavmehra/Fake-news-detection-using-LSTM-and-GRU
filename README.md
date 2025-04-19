Fake News Detection using Bi-directional LSTM & GRU

This project implements a deep learning approach for detecting fake news articles using NLP techniques and neural sequence models. The goal is to classify news articles as either fake or real using the ISOT Fake News Dataset, achieving over 99% accuracy using Bi-directional LSTM and GRU architectures.

-----------------------------------
📊 Dataset Used: ISOT Fake News Dataset
-----------------------------------
The dataset consists of two files:
- `True.csv`: real news articles
- `Fake.csv`: fake news articles

Each article contains:
- `title`
- `text` (full article content)
- `subject`
- `date`

A binary `label` column was added:  
- `0` → Real  
- `1` → Fake

-----------------------------------
🔄 Preprocessing Approach
-----------------------------------
1. **Text Cleaning:**
   - Combined `title` and `text` into one column
   - Converted all text to lowercase
   - Removed punctuation, digits, and special characters
   - Removed extra whitespace

2. **Stopword Removal:**
   - Removed common English stopwords using NLTK

3. **Lemmatization:**
   - Used `spaCy` (`en_core_web_sm`) to lemmatize each word
   - Retained only meaningful root forms (e.g., "running" → "run")

4. **Normalization:**
   - Removed character elongations (e.g., "cooool" → "cool")
   - Removed repeated whitespaces

5. **Tokenization & Padding:**
   - Tokenized text using Keras `Tokenizer`
   - Padded all sequences to the same maximum length (300)

-----------------------------------
💡 Embedding & Modeling Approach
-----------------------------------
1. **Text Vectorization:**
   - Used Keras’ `Tokenizer` to convert text to integer sequences
   - Used `Embedding` layer (embedding_dim=128) to convert tokens into dense vectors

2. **Model Architectures:**
   - Bi-directional LSTM
   - Bi-directional GRU

3. **Model Layers:**
   - Embedding Layer (non-trainable)
   - Bidirectional LSTM / GRU
   - Dense layer with ReLU + Dropout
   - Output layer with Sigmoid activation

4. **Training:**
   - Optimizer: Adam
   - Loss: Binary Crossentropy
   - Metrics: Accuracy
   - EarlyStopping callback to prevent overfitting

-----------------------------------
📈 Performance
-----------------------------------
| Model           | Accuracy | Loss |
|-----------------|----------|------|
| BiLSTM          | ~99%     | Low  |
| BiGRU           | ~99%     | Low  |

Evaluation metrics:
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- Accuracy & Loss Plots

-----------------------------------
🚀 How to Run (Google Colab Recommended)
-----------------------------------
1. Preprocess the data using the provided preprocessing pipeline
2. Train BiLSTM and BiGRU models on the processed data
3. Evaluate using test set metrics
4. Visualize training and validation curves

-----------------------------------
📚 Libraries Used
-----------------------------------
- Python 3.x
- TensorFlow / Keras
- Pandas / NumPy / Matplotlib / Seaborn
- Scikit-learn
- NLTK / spaCy (`en_core_web_sm`)

-----------------------------------
📌 Project Status: Completed
🎯 Final Accuracy: Over 99%
