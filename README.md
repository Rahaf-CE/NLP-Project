🩺 Arabic Medical Question Answering System
Using Traditional IR Methods and Transformer-Based Models
         
🔍 Dataset
Source: Kaggle – Arabic Medical Q&A Dataset

Contains thousands of Arabic medical Q&A pairs.

Topics: Tumors, Diabetes, Endocrinology, Hypertension, Surgery, etc.

After preprocessing: cleaned, normalized, diacritics removed, and labels standardized.

Split: 70% training, 15% validation, 15% testing.

🛠️ Preprocessing Steps
Normalization of Arabic letters (e.g., all forms of Alif unified).

Diacritics removal to reduce token variation.

Stopwords removal using NLTK Arabic list.

Lemmatization using Qalsadi lemmatizer.

Tokenization with NLTK.

🧠 Models Used
Model	Description
TF-IDF + Cosine	Classic baseline, vectorizes text with term frequency-inverse document frequency
BM25	Probabilistic IR model, strong lexical retrieval
SBERT	Pre-trained multilingual Sentence-BERT, semantic similarity
AraBERT	Arabic-specific BERT transformer model

📈 Evaluation Metrics
Top-1 Accuracy

Recall@5

Precision@5

F1-Score

BM25 outperformed all models in all metrics, but AraBERT and SBERT showed strong semantic understanding.


🔮 Future Work
Use cross-encoders for better semantic matching.

Involve medical experts for manual evaluation of answer quality.

Integrate AraGPT or RAG (Retrieval-Augmented Generation) models.

Expand the dataset to cover pharmaceutical and psychological consultations.

Combine traditional + neural methods for hybrid retrieval systems.

📚 References
[1] Wolf et al., “Transformers: State-of-the-Art NLP”, EMNLP 2020

[2] Reimers & Gurevych, “Sentence-BERT”, arXiv:1908.10084

[3] Antoun et al., “AraBERT”, Workshop on Open-Source Arabic NLP, 2020

[4] Kaggle – Arabic Medical Q&A Dataset

[7] Robertson & Zaragoza, “BM25 and Beyond”, 2009

📦 Installation & Usage
bash
Copy
Edit
cd arabic-medical-qa
pip install -r requirements.txt
To test a model (e.g., BM25):

bash
Copy
Edit
python main.py --model bm25 --question "ما هي أسباب ارتفاع ضغط الدم؟"
✅ Requirements
Python 3.8+

transformers

sentence-transformers

scikit-learn

rank_bm25

nltk

qalsadi

pandas, numpy, matplotlib, seaborn
