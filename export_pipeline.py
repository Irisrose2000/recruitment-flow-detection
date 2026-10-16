import os
import re
import json
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+|#URL_\w+#', ' ', text)
    text = re.sub(r'#PHONE_\w+#|#EMAIL_\w+#', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = text.split()
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(cleaned_tokens)

def main():
    os.makedirs('models', exist_ok=True)
    print("Loading dataset...")
    df = pd.read_csv('archive/DataSet.csv')
    df['label'] = df['fraudulent'].map({'t': 1, 'f': 0}).fillna(0).astype(int)

    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        df[col] = df[col].fillna('')

    df['combined_text'] = (
        df['title'] + " " +
        df['company_profile'] + " " +
        df['description'] + " " +
        df['requirements'] + " " +
        df['benefits']
    )

    print("Cleaning text...")
    df['cleaned_text'] = df['combined_text'].apply(clean_text)

    # Stratified split
    X_train_raw, X_test_raw, y_train, y_test, idx_train, idx_test = train_test_split(
        df['cleaned_text'], df['label'], df.index, test_size=0.20, random_state=42, stratify=df['label']
    )

    print("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train_raw)

    print("Training Logistic Regression (Balanced)...")
    lr_balanced = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_balanced.fit(X_train_tfidf, y_train)

    print("Training Logistic Regression (Default)...")
    lr_default = LogisticRegression(max_iter=1000, random_state=42)
    lr_default.fit(X_train_tfidf, y_train)

    print("Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_tfidf, y_train)

    print("Saving models to disk...")
    joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib')
    joblib.dump(lr_balanced, 'models/lr_balanced.joblib')
    joblib.dump(lr_default, 'models/lr_default.joblib')
    joblib.dump(nb_model, 'models/naive_bayes.joblib')

    # Select representative samples from test set for quick testing
    test_df = df.loc[idx_test]
    fraud_candidates = test_df[test_df['label'] == 1].head(3)
    legit_candidates = test_df[test_df['label'] == 0].head(3)

    samples = {
        "Scam: High-Paying Data Entry Clerk (No Experience)": (
            "URGENT HIRING: Data Entry Operator / Payroll Assistant.\n"
            "We are seeking motivated individuals to work from home with flexible hours. "
            "No prior experience or education required. Earn $45-$65 per hour plus weekly bonuses! "
            "Responsibilities include entering order data into our remote portal, managing wire transfers, "
            "and verifying customer invoices. We provide all equipment: you will receive a cashier check to purchase "
            "home office supplies through our designated vendor. Wire remaining funds via Western Union or MoneyGram. "
            "To apply, send your full legal name, personal email, phone number, and banking institution name to "
            "careers-global-hiring@gmail.com or contact us directly on Telegram @fast_payroll_hr. Immediate placement guaranteed."
        ),
        "Authentic Fraud Example from Dataset (Administrative Assistant)": (
            df.loc[test_df[test_df['label'] == 1].index[0], 'combined_text'][:1200]
        ),
        "Legitimate: Senior Software Engineer (Python/Cloud)": (
            "Senior Backend Software Engineer (Python / Distributed Systems)\n"
            "About Acme Cloud: We are a Series B funded infrastructure technology company based in San Francisco, CA. "
            "We build high-performance data processing pipelines that handle billions of events daily for enterprise clients worldwide.\n"
            "Key Responsibilities:\n"
            "- Architect and scale distributed microservices using Python, FastAPI, and Go on AWS/GCP.\n"
            "- Collaborate with cross-functional product and DevOps teams to ensure 99.99% uptime and low latency.\n"
            "- Implement automated testing, CI/CD pipelines, and robust monitoring with Prometheus and Grafana.\n"
            "Requirements:\n"
            "- 5+ years of production experience in backend software engineering with Python.\n"
            "- Strong understanding of relational databases (PostgreSQL) and caching layers (Redis).\n"
            "- Bachelor's or Master's degree in Computer Science, Software Engineering, or equivalent practical experience.\n"
            "Benefits:\n"
            "- Competitive base salary ($150,000 - $185,000) with equity and 401(k) matching.\n"
            "- Comprehensive health, dental, and vision insurance.\n"
            "- Annual learning stipend and flexible paid time off."
        ),
        "Authentic Legitimate Example from Dataset (Customer Success)": (
            df.loc[test_df[test_df['label'] == 0].index[0], 'combined_text'][:1200]
        )
    }

    with open('models/sample_jobs.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)

    print("Pipeline and models exported successfully!")

if __name__ == '__main__':
    main()
