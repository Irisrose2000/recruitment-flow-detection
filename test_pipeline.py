import os
import re
import sys
import json
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Setup resources
try:
    stops = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stops = set(stopwords.words('english'))

try:
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize('test')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()

def clean_input_text(text):
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
        if token not in stops and len(token) > 2
    ]
    return " ".join(cleaned_tokens)

def scan_risk_indicators(text):
    text_lower = text.lower()
    indicators = []
    
    patterns = [
        {
            'category': 'Financial Red Flag',
            'pattern': r'\b(wire transfer|western union|moneygram|cashier\s*check|send money|pay fee|advance fee|deposit\s*check|cryptocurrency|bitcoin)\b',
            'severity': 'HIGH',
            'detail': 'Mentions wire transfers, check cashing, advance fees, or cryptocurrency.',
            'tip': 'Legitimate employers never ask employees to deposit checks or wire funds for supplies.'
        },
        {
            'category': 'Unrealistic Compensation / No Experience',
            'pattern': r'\b(no experience (required|needed)|earn \$[4-9]\d{1,2}|earn \$[1-9]\d{3}|no interview|make money fast|instant cash)\b',
            'severity': 'MEDIUM',
            'detail': 'Promises very high pay with no prior experience or screening.',
            'tip': 'Unusually high compensation for entry-level work is a classic bait tactic.'
        },
        {
            'category': 'Unprofessional Contact Channel',
            'pattern': r'(@gmail\.com|@yahoo\.com|@hotmail\.com|telegram|whatsapp|google hangouts|text me on)\b',
            'severity': 'HIGH',
            'detail': 'Directs communication to free webmail (Gmail/Yahoo) or messaging apps (Telegram/WhatsApp).',
            'tip': 'Reputable hiring managers communicate via official corporate email domains.'
        },
        {
            'category': 'Urgent Placement Pressure',
            'pattern': r'\b(urgent hiring|immediate placement|start today|urgent vacancy|apply immediately|limited spots)\b',
            'severity': 'LOW',
            'detail': 'Appeals to urgency to rush candidates through normal due diligence.',
            'tip': 'Scammers use false urgency so victims make hasty decisions without verifying legitimacy.'
        },
        {
            'category': 'Sensitive Personal Info Request',
            'pattern': r'\b(social security|ssn|bank account details|routing number|credit card|date of birth)\b',
            'severity': 'HIGH',
            'detail': 'Solicits sensitive banking or identity credentials upfront.',
            'tip': 'Never provide banking credentials or SSN before receiving a formal, verified written offer.'
        }
    ]
    
    for item in patterns:
        matches = re.findall(item['pattern'], text_lower)
        if matches:
            indicators.append({
                'category': item['category'],
                'severity': item['severity'],
                'detail': item['detail'],
                'tip': item['tip'],
                'matched': list(set(matches[:3]))
            })
            
    return indicators

def evaluate_posting(raw_text, tfidf, models, threshold=0.50):
    cleaned = clean_input_text(raw_text)
    vec = tfidf.transform([cleaned])
    
    prob_bal = float(models['Logistic Regression (Balanced)'].predict_proba(vec)[0][1])
    prob_std = float(models['Logistic Regression (Standard)'].predict_proba(vec)[0][1])
    prob_nb  = float(models['Multinomial Naive Bayes'].predict_proba(vec)[0][1])
    
    flags = scan_risk_indicators(raw_text)
    has_high_flag = any(f['severity'] in ['HIGH', 'CRITICAL'] for f in flags)
    
    # Unified Assessment Logic
    if prob_bal >= threshold or has_high_flag:
        risk_level = "HIGH RISK"
        verdict = "SUSPECTED SCAM"
    elif prob_bal >= 0.25 or len(flags) > 0:
        risk_level = "MODERATE RISK"
        verdict = "PROCEED WITH CAUTION"
    else:
        risk_level = "LOW RISK"
        verdict = "LIKELY LEGITIMATE"
        
    return {
        'prob_bal': prob_bal,
        'prob_std': prob_std,
        'prob_nb': prob_nb,
        'flags': flags,
        'risk_level': risk_level,
        'verdict': verdict
    }

def run_suite():
    print("=" * 85)
    print("JOB FRAUD SHIELD: AUTOMATED TEST SUITE")
    print("Evaluating Legitimate, Suspicious, and Ambiguous Benchmark Cases")
    print("=" * 85)
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    tfidf = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    models = {
        'Logistic Regression (Balanced)': joblib.load(os.path.join(models_dir, 'lr_balanced.joblib')),
        'Logistic Regression (Standard)': joblib.load(os.path.join(models_dir, 'lr_default.joblib')),
        'Multinomial Naive Bayes': joblib.load(os.path.join(models_dir, 'naive_bayes.joblib'))
    }
    
    with open(os.path.join(models_dir, 'sample_jobs.json'), 'r', encoding='utf-8') as f:
        samples = json.load(f)
        
    results = []
    
    for name, text in samples.items():
        if "Legitimate:" in name:
            cat = "Legitimate"
            expected_verdict = "LIKELY LEGITIMATE"
        elif "Suspicious:" in name or "Scam:" in name:
            cat = "Suspicious"
            expected_verdict = "SUSPECTED SCAM"
        else:
            cat = "Ambiguous"
            expected_verdict = "CAUTION / INVESTIGATION"
            
        res = evaluate_posting(text, tfidf, models)
        
        # Test status validation
        if cat == "Legitimate":
            passed = (res['verdict'] == "LIKELY LEGITIMATE") and (len(res['flags']) == 0)
        elif cat == "Suspicious":
            passed = (res['verdict'] == "SUSPECTED SCAM") and (len(res['flags']) > 0 or res['prob_bal'] >= 0.50)
        else: # Ambiguous
            # Ambiguous postings must be detected with elevated probabilities or cautionary flags
            passed = (res['risk_level'] in ["MODERATE RISK", "HIGH RISK"])
            
        status = "PASSED" if passed else "FAILED"
        
        results.append({
            'Posting': name[:45] + '...',
            'Category': cat,
            'LR Balanced': f"{res['prob_bal']*100:5.1f}%",
            'Naive Bayes': f"{res['prob_nb']*100:5.1f}%",
            'LR Standard': f"{res['prob_std']*100:5.1f}%",
            'Flags': len(res['flags']),
            'Assessed Level': res['risk_level'],
            'System Verdict': res['verdict'],
            'Status': status
        })
        
        print(f"[{status}] {cat.upper()}: {name}")
        print(f"       LR Balanced: {res['prob_bal']*100:.1f}% | Naive Bayes: {res['prob_nb']*100:.1f}% | LR Standard: {res['prob_std']*100:.1f}%")
        print(f"       Flags: {len(res['flags'])} -> {[f['category'] for f in res['flags']]}")
        print(f"       Verdict: {res['verdict']} ({res['risk_level']})\n")

    df_summary = pd.DataFrame(results)
    print("=" * 85)
    print("SUMMARY RESULTS MATRIX:")
    print("=" * 85)
    print(df_summary[['Posting', 'Category', 'LR Balanced', 'Flags', 'Assessed Level', 'Status']].to_string(index=False))
    
    df_summary.to_csv("test_results_summary.csv", index=False)
    print("\nSaved detailed results to test_results_summary.csv")
    
    all_passed = all(r['Status'] == 'PASSED' for r in results)
    print(f"\nFinal Test Status: {'ALL TESTS PASSED SUCCESSFULLY (8/8)' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed

if __name__ == '__main__':
    ok = run_suite()
    if not ok:
        sys.exit(1)
