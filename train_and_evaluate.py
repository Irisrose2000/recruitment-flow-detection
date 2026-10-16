import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Ensure NLTK resources are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize('testing')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()

# Define cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs and URL tokens like #URL_...#
    text = re.sub(r'http\S+|www\.\S+|#URL_\w+#', ' ', text)
    # Remove phone / email hashes
    text = re.sub(r'#PHONE_\w+#|#EMAIL_\w+#', ' ', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase
    text = text.lower()
    # Tokenize by whitespace, remove short words & stopwords, apply lemmatization
    tokens = text.split()
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(cleaned_tokens)

def load_and_preprocess_data(csv_path="archive/DataSet.csv"):
    print("1. Loading dataset from:", csv_path)
    df = pd.read_csv(csv_path)
    print(f"   Original dataset shape: {df.shape}")
    
    # Target variable conversion: 't' -> 1 (fraudulent), 'f' -> 0 (legitimate)
    if df['fraudulent'].dtype == object:
        df['label'] = df['fraudulent'].map({'t': 1, 'f': 0})
    else:
        df['label'] = df['fraudulent'].astype(int)
        
    print(f"   Target distribution:\n{df['label'].value_counts(normalize=True).map(lambda p: f'{p:.2%}')}")
    print(f"   Class counts: 0 (Legitimate): {(df['label'] == 0).sum()}, 1 (Fraudulent): {(df['label'] == 1).sum()}")

    # Combine informative text columns
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        df[col] = df[col].fillna('')

    print("2. Combining text fields (title, company_profile, description, requirements, benefits)...")
    df['combined_text'] = (
        df['title'] + " " +
        df['company_profile'] + " " +
        df['description'] + " " +
        df['requirements'] + " " +
        df['benefits']
    )

    print("3. Cleaning text (stripping HTML, URLs, symbols, stopwords, and lemmatizing)...")
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Filter out empty texts if any
    non_empty = df['cleaned_text'].str.strip() != ''
    df = df[non_empty].reset_index(drop=True)
    print(f"   Processed {len(df)} non-empty rows.")
    
    return df

def plot_confusion_matrices(cm_dict, output_path="confusion_matrices.png"):
    fig, axes = plt.subplots(1, len(cm_dict), figsize=(6 * len(cm_dict), 5))
    if len(cm_dict) == 1:
        axes = [axes]
        
    classes = ['Legitimate (0)', 'Fraudulent (1)']
    
    for ax, (model_name, cm) in zip(axes, cm_dict.items()):
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Confusion Matrix:\n{model_name}', fontsize=12, fontweight='bold', pad=12)
        
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)
        
        # Annotate counts and percentages
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cell_text = f"{cm[i, j]:,}\n({cm[i, j] / cm.sum() * 100:.1f}%)"
                ax.text(j, i, cell_text,
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11, fontweight='medium')
                
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Confusion matrices saved to {output_path}")

def plot_metrics_comparison(metrics_df, output_path="metrics_comparison.png"):
    metrics = ['Precision (Fraud)', 'Recall (Fraud)', 'F1-Score (Fraud)', 'Macro F1', 'Accuracy']
    x = np.arange(len(metrics))
    width = 0.8 / len(metrics_df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, (model_name, row) in enumerate(metrics_df.iterrows()):
        vals = [row[m] for m in metrics]
        rects = ax.bar(x + idx * width, vals, width, label=model_name, color=colors[idx % len(colors)])
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0)

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison across Models', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width * (len(metrics_df) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Metrics comparison chart saved to {output_path}")

def main():
    # 1. Load and Clean
    df = load_and_preprocess_data("archive/DataSet.csv")
    
    X = df['cleaned_text']
    y = df['label']
    
    # 2. Train-Test Split (Stratified)
    print("\n4. Splitting data into training (80%) and testing (20%) sets (stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Training samples: {len(X_train_raw)} (Fraud: {y_train.sum()}, Legitimate: {len(y_train) - y_train.sum()})")
    print(f"   Testing samples:  {len(X_test_raw)} (Fraud: {y_test.sum()}, Legitimate: {len(y_test) - y_test.sum()})")

    # 3. TF-IDF Vectorization
    print("\n5. Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    print(f"   TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")

    # 4. Train Models
    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
        'Logistic Regression (Default)': LogisticRegression(max_iter=1000, random_state=42),
        'Logistic Regression (Balanced)': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    }

    results = {}
    cm_dict = {}

    print("\n6. Training and Evaluating Models...")
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        y_prob = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        prec_fraud = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_fraud = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_fraud = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        cm = confusion_matrix(y_test, y_pred)
        
        cm_dict[name] = cm
        results[name] = {
            'Accuracy': acc,
            'Precision (Fraud)': prec_fraud,
            'Recall (Fraud)': rec_fraud,
            'F1-Score (Fraud)': f1_fraud,
            'Macro Precision': prec_macro,
            'Macro Recall': rec_macro,
            'Macro F1': f1_macro,
            'ROC-AUC': auc,
            'TN': cm[0, 0],
            'FP': cm[0, 1],
            'FN': cm[1, 0],
            'TP': cm[1, 1]
        }
        
        print(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraudulent (1)'], digits=4)}")
        print(f"Confusion Matrix:\n{cm}")

    # Create summary DataFrame
    metrics_df = pd.DataFrame(results).T
    print("\n=== SUMMARY METRICS TABLE ===")
    print(metrics_df[['Accuracy', 'Precision (Fraud)', 'Recall (Fraud)', 'F1-Score (Fraud)', 'Macro F1', 'ROC-AUC']].to_string())
    
    # Save results to CSV
    metrics_df.to_csv("model_comparison_results.csv", index=True)
    print("\nSaved metrics to model_comparison_results.csv")

    # Generate Plots
    print("\n7. Generating visualizations...")
    plot_confusion_matrices(cm_dict, "confusion_matrices.png")
    plot_metrics_comparison(metrics_df, "metrics_comparison.png")
    print("Done!")

if __name__ == '__main__':
    main()
