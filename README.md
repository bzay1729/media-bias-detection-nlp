# Media Bias Detection using NLP
NLP-based media bias detection system using machine learning and transformer models to classify political and media bias in news text.

# Overview
This project implements an NLP-based media bias detection system designed to classify bias in news headlines and articles. The goal is to identify ideological or political bias using traditional machine learning and transformer-based deep learning models.
The system focuses on:
1. Text preprocessing
2. Feature engineering
3. Model training
4.  Evaluation
while addressing real-world challenges such as class imbalance and subjective labeling in media bias datasets.

# bjectives
Detect and classify media bias from news text
Compare traditional ML models with transformer-based architectures
Address class imbalance using data-level techniques
Evaluate models using robust classification metrics

# Datasets
News headlines and article text labeled for media bias.
Labels represent different bias categories such as left, center, right leaning.
Data preprocessing includes:
  a. Text normalization
  b. Tokenization
  c. Stopword removal
  d. Label encoding
Dataset files may be excluded due to licensing or size constraints.

# Models Implemented
1. Traditional Machine Learning
2. Logistic Regression
3. Random Forest
4. Deep Learning / Transformers
5. MLP (Neural Network)
6. CNN (Text-based)
7. Transformer models (RoBERTa / PoliBERT / XLM-RoBERTa)
   
# Techniques Used
- Text preprocessing & normalization
- TF-IDF and embedding-based representations
- SMOTE for class imbalance handling
- Hyperparameter tuning
- Model benchmarking and comparison

# Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
Special emphasis was placed on recall and F1-score due to imbalance in bias classes.

# Key Results
- Transformer-based models significantly outperformed traditional ML approaches.
- SMOTE improved minority-class recall at the cost of slight precision trade-offs.
- Fine-tuned transformer models achieved the best overall generalization.

# How to Run
## 1. Clone the repository
git clone https://github.com/your-username/media-bias-detection-nlp.git
cd media-bias-detection-nlp
## 2. Install dependencies
pip install -r requirements.txt
## 3. Run training
python src/train.py
## 4. Evaluate model
python src/evaluate.py

# Tools & Technologies
- Python
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Pandas, NumPy
- Matplotlib / Seaborn
  
# 👤 Author
Bijay Pokhrel
M.S. Computer Science Specialization AI/ML
Southern Methodist University
