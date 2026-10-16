"""
Job Fraud Shield - NLP Model Module
===================================
Develops an NLP-based job-posting classifier using TF-IDF and Logistic Regression,
with Multinomial Naive Bayes as a baseline for comparison.

Includes:
- Robust text cleaning, tokenization, stopword removal, and lemmatization.
- TF-IDF n-gram vectorization (unigrams & bigrams).
- Primary Model: Logistic Regression (Balanced class weights for handling severe imbalance).
- Baseline Model: Multinomial Naive Bayes (alpha=0.1).
- Secondary Benchmark: Standard Logistic Regression.
- Model serialization (joblib) and inference API with feature attribution explainability.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------------------------
# 1. Text Preprocessing & Cleaning Pipeline
# ------------------------------------------------------------------------------
class TextPreprocessor:
    """Handles text cleaning, tokenization, stopword removal, and lemmatization."""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
            
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.lemmatizer.lemmatize('test')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            self.lemmatizer = WordNetLemmatizer()
            
    def clean(self, text: str) -> str:
        """Cleans raw job posting text."""
        if not isinstance(text, str):
            return ""
            
        # Strip HTML markup
        text = re.sub(r'<[^>]+>', ' ', text)
        # Strip URLs and dataset URL tokens like #URL_...#
        text = re.sub(r'http\S+|www\.\S+|#URL_\w+#', ' ', text)
        # Strip phone and email anonymization hashes
        text = re.sub(r'#PHONE_\w+#|#EMAIL_\w+#', ' ', text)
        # Keep alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Lowercase
        text = text.lower()
        
        # Tokenize, remove stopwords, filter short words, and lemmatize
        tokens = text.split()
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        return " ".join(cleaned_tokens)


# ------------------------------------------------------------------------------
# 2. End-to-End Model Pipeline
# ------------------------------------------------------------------------------
class JobFraudModelPipeline:
    """
    NLP Job Posting Classification Pipeline.
    Trained on TF-IDF features using Logistic Regression with Naive Bayes baseline.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.preprocessor = TextPreprocessor()
        self.tfidf: Optional[TfidfVectorizer] = None
        self.primary_model: Optional[LogisticRegression] = None   # Balanced LR
        self.baseline_model: Optional[MultinomialNB] = None       # Naive Bayes
        self.standard_model: Optional[LogisticRegression] = None   # Default LR
        
    def train(
        self,
        csv_path: str = "archive/DataSet.csv",
        test_size: float = 0.20,
        random_state: int = 42,
        max_features: int = 10000
    ) -> pd.DataFrame:
        """
        Loads dataset, performs stratified train-test split, fits TF-IDF vectorizer,
        trains Balanced Logistic Regression and Baseline Naive Bayes, and evaluates.
        """
        print(f"[1/5] Loading raw dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"      Total records loaded: {len(df):,}")
        
        # Encode target: 't' -> 1 (fraudulent), 'f' -> 0 (legitimate)
        if df['fraudulent'].dtype == object:
            df['label'] = df['fraudulent'].map({'t': 1, 'f': 0}).fillna(0).astype(int)
        else:
            df['label'] = df['fraudulent'].astype(int)
            
        legit_cnt = int((df['label'] == 0).sum())
        fraud_cnt = int((df['label'] == 1).sum())
        print(f"      Class distribution: Legitimate = {legit_cnt:,} ({legit_cnt/len(df)*100:.2f}%), Fraudulent = {fraud_cnt:,} ({fraud_cnt/len(df)*100:.2f}%)")
        print(f"      Imbalance Ratio: {legit_cnt/fraud_cnt:.1f} : 1")

        # Combine informative text columns
        text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
        for col in text_cols:
            df[col] = df[col].fillna('')
            
        print("[2/5] Combining text fields & applying preprocessing (cleaning, tokenizing, lemmatizing)...")
        combined_text = (
            df['title'] + " " +
            df['company_profile'] + " " +
            df['description'] + " " +
            df['requirements'] + " " +
            df['benefits']
        )
        df['cleaned_text'] = combined_text.apply(self.preprocessor.clean)
        
        # Filter non-empty rows
        valid_mask = df['cleaned_text'].str.strip() != ''
        df = df[valid_mask].reset_index(drop=True)
        print(f"      Cleaned dataset retained {len(df):,} non-empty rows.")

        # Stratified train-test split
        print(f"[3/5] Performing stratified train-test split ({int((1-test_size)*100)}% train / {int(test_size*100)}% test)...")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label'],
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )
        print(f"      Train set: {len(X_train_raw):,} postings ({y_train.sum()} fraud)")
        print(f"      Test set:  {len(X_test_raw):,} postings ({y_test.sum()} fraud)")

        # TF-IDF Vectorization
        print(f"[4/5] Fitting TF-IDF Vectorizer (max_features={max_features}, ngram_range=(1,2))...")
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
        X_train_tfidf = self.tfidf.fit_transform(X_train_raw)
        X_test_tfidf = self.tfidf.transform(X_test_raw)
        print(f"      Vocabulary size: {len(self.tfidf.vocabulary_):,} n-grams")

        # Define Models
        models = {
            'Logistic Regression (Balanced)': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=random_state
            ),
            'Multinomial Naive Bayes (Baseline)': MultinomialNB(alpha=0.1),
            'Logistic Regression (Standard)': LogisticRegression(
                max_iter=1000,
                random_state=random_state
            )
        }

        print("[5/5] Training models and evaluating on held-out test data...")
        results = {}
        for name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            y_prob = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'Accuracy': acc,
                'Precision (Fraud)': prec,
                'Recall (Fraud)': rec,
                'F1-Score (Fraud)': f1,
                'ROC-AUC': auc,
                'True Positives (Caught)': int(cm[1, 1]),
                'False Negatives (Missed)': int(cm[1, 0]),
                'False Positives (Alarms)': int(cm[0, 1]),
                'True Negatives (Legit)': int(cm[0, 0])
            }

        self.primary_model = models['Logistic Regression (Balanced)']
        self.baseline_model = models['Multinomial Naive Bayes (Baseline)']
        self.standard_model = models['Logistic Regression (Standard)']

        metrics_df = pd.DataFrame(results).T
        print("\n" + "=" * 90)
        print("MODEL EVALUATION & BENCHMARK MATRIX:")
        print("=" * 90)
        print(metrics_df[['Accuracy', 'Precision (Fraud)', 'Recall (Fraud)', 'F1-Score (Fraud)', 'ROC-AUC', 'True Positives (Caught)', 'False Negatives (Missed)']].to_string())
        print("=" * 90)

        return metrics_df

    def save(self, output_dir: Optional[str] = None):
        """Saves trained vectorizer and models to disk."""
        save_dir = output_dir or self.models_dir
        os.makedirs(save_dir, exist_ok=True)
        
        joblib.dump(self.tfidf, os.path.join(save_dir, 'tfidf_vectorizer.joblib'))
        joblib.dump(self.primary_model, os.path.join(save_dir, 'lr_balanced.joblib'))
        joblib.dump(self.baseline_model, os.path.join(save_dir, 'naive_bayes.joblib'))
        joblib.dump(self.standard_model, os.path.join(save_dir, 'lr_default.joblib'))
        print(f"Serialized models successfully to: {save_dir}/")

    def load(self, load_dir: Optional[str] = None):
        """Loads trained vectorizer and models from disk."""
        target_dir = load_dir or self.models_dir
        self.tfidf = joblib.load(os.path.join(target_dir, 'tfidf_vectorizer.joblib'))
        self.primary_model = joblib.load(os.path.join(target_dir, 'lr_balanced.joblib'))
        self.standard_model = joblib.load(os.path.join(target_dir, 'lr_default.joblib'))
        self.baseline_model = joblib.load(os.path.join(target_dir, 'naive_bayes.joblib'))
        print(f"Loaded models successfully from: {target_dir}/")

    def predict(self, raw_text: str, threshold: float = 0.50, model_type: str = 'primary') -> Dict[str, Any]:
        """
        Runs full inference on a raw job posting string.
        model_type: 'primary' (Balanced LR), 'baseline' (Naive Bayes), or 'standard' (Default LR)
        """
        if self.tfidf is None:
            raise ValueError("Model pipeline is not fitted or loaded yet. Call load() or train().")
            
        model_map = {
            'primary': self.primary_model,
            'baseline': self.baseline_model,
            'standard': self.standard_model
        }
        model = model_map.get(model_type, self.primary_model)
        
        cleaned = self.preprocessor.clean(raw_text)
        if not cleaned:
            return {
                'error': 'Text contains insufficient alphabetic tokens after cleaning.',
                'fraud_probability': 0.0,
                'is_fraud': False,
                'risk_level': 'UNKNOWN'
            }
            
        vec = self.tfidf.transform([cleaned])
        prob_fraud = float(model.predict_proba(vec)[0][1])
        prob_legit = float(model.predict_proba(vec)[0][0])
        is_fraud = prob_fraud >= threshold
        
        if is_fraud:
            risk_level = "HIGH RISK (SUSPECTED SCAM)"
        elif prob_fraud >= 0.30:
            risk_level = "MODERATE RISK (PROCEED WITH CAUTION)"
        else:
            risk_level = "LOW RISK (LIKELY LEGITIMATE)"

        return {
            'fraud_probability': round(prob_fraud, 4),
            'legitimate_probability': round(prob_legit, 4),
            'decision_threshold': threshold,
            'is_fraud': is_fraud,
            'risk_level': risk_level,
            'model_used': type(model).__name__
        }

    def explain(self, raw_text: str, top_k: int = 8) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Extracts token contributions for the input text based on Logistic Regression weights.
        """
        if self.tfidf is None or self.primary_model is None:
            return None
            
        cleaned = self.preprocessor.clean(raw_text)
        if not cleaned:
            return None
            
        feature_names = np.array(self.tfidf.get_feature_names_out())
        tfidf_vec = self.tfidf.transform([cleaned])
        nonzero_indices = tfidf_vec.nonzero()[1]
        
        if len(nonzero_indices) == 0:
            return None
            
        weights = self.primary_model.coef_[0][nonzero_indices]
        tfidf_vals = tfidf_vec.data
        contributions = weights * tfidf_vals
        terms = feature_names[nonzero_indices]
        
        df_contrib = pd.DataFrame({
            'term': terms,
            'contribution': contributions,
            'weight': weights,
            'tfidf': tfidf_vals
        })
        
        fraud_leaning = df_contrib[df_contrib['contribution'] > 0].sort_values(by='contribution', ascending=False).head(top_k)
        legit_leaning = df_contrib[df_contrib['contribution'] < 0].sort_values(by='contribution', ascending=True).head(top_k)
        
        return {
            'fraud_leaning': fraud_leaning,
            'legit_leaning': legit_leaning
        }


# ------------------------------------------------------------------------------
# 3. Main Script Execution
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    pipeline = JobFraudModelPipeline()
    csv_file = "archive/DataSet.csv"
    
    if os.path.exists(csv_file):
        print("Training Job Fraud Shield Model Pipeline...")
        metrics = pipeline.train(csv_path=csv_file)
        pipeline.save()
        
        # Test sample inference
        sample_scam = (
            "URGENT HIRING: Data Entry Operator. Earn $55/hr working from home! No experience required. "
            "A company cashier check will be sent to your address. Deposit check and wire transfer balance "
            "via Western Union to our equipment vendor. Contact Telegram @fast_hire."
        )
        print("\n--- SAMPLE INFERENCE TEST ---")
        pred = pipeline.predict(sample_scam)
        print("Inference Result:", pred)
        
        expl = pipeline.explain(sample_scam)
        if expl and not expl['fraud_leaning'].empty:
            print("\nTop Contributing Scam Tokens:")
            print(expl['fraud_leaning'][['term', 'contribution', 'weight']])
    else:
        print(f"Dataset {csv_file} not found. Attempting to load existing serialized models...")
        pipeline.load()
        print("Models ready for inference.")
