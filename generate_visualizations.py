import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Professional clean styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#D1D5DB'
plt.rcParams['axes.linewidth'] = 0.8

def analyze_and_generate():
    print("Step 1: Loading raw dataset...")
    df = pd.read_csv('archive/DataSet.csv')
    df['label'] = df['fraudulent'].map({'t': 1, 'f': 0}).fillna(0).astype(int)
    
    total_records = len(df)
    fraud_count = int((df['label'] == 1).sum())
    legit_count = int((df['label'] == 0).sum())
    fraud_pct = fraud_count / total_records * 100
    legit_pct = legit_count / total_records * 100

    print(f"Total: {total_records}, Legit: {legit_count} ({legit_pct:.2f}%), Fraud: {fraud_count} ({fraud_pct:.2f}%)")

    # Metadata flags
    metadata_flags = {}
    for col in ['has_company_logo', 'telecommuting', 'has_questions']:
        s_legit = (df[df['label'] == 0][col].astype(str).str.lower().isin(['t', 'true', '1'])).mean() * 100
        s_fraud = (df[df['label'] == 1][col].astype(str).str.lower().isin(['t', 'true', '1'])).mean() * 100
        metadata_flags[col] = {
            'legitimate': round(float(s_legit), 2),
            'fraudulent': round(float(s_fraud), 2)
        }

    # Missing rates
    missing_rates = {}
    for col in ['company_profile', 'requirements', 'benefits', 'salary_range', 'department']:
        legit_null = float(df[df['label'] == 0][col].isna().mean() * 100)
        fraud_null = float(df[df['label'] == 1][col].isna().mean() * 100)
        missing_rates[col] = {
            'legitimate': round(legit_null, 2),
            'fraudulent': round(fraud_null, 2)
        }

    # Text length stats
    text_lengths = {}
    for col in ['description', 'requirements', 'company_profile']:
        series = df[col].fillna('')
        words = series.apply(lambda x: len(x.split()))
        text_lengths[col] = {
            'legit_avg_words': round(float(words[df['label'] == 0].mean()), 1),
            'fraud_avg_words': round(float(words[df['label'] == 1].mean()), 1),
            'legit_median_words': round(float(words[df['label'] == 0].median()), 1),
            'fraud_median_words': round(float(words[df['label'] == 1].median()), 1)
        }

    # Contact tokens
    for col in ['title', 'company_profile', 'description', 'requirements', 'benefits']:
        df[col] = df[col].fillna('')
    combined_raw = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits']
    
    token_stats = {}
    for token in ['#EMAIL_', '#PHONE_', '#URL_']:
        has_token = combined_raw.str.contains(token)
        token_stats[token] = {
            'legitimate': round(float(has_token[df['label'] == 0].mean() * 100), 2),
            'fraudulent': round(float(has_token[df['label'] == 1].mean() * 100), 2)
        }

    # Top fraud industries & functions
    top_fraud_industries = df[df['label'] == 1]['industry'].dropna().value_counts().head(8).to_dict()
    top_fraud_functions = df[df['label'] == 1]['function'].dropna().value_counts().head(8).to_dict()
    top_fraud_titles = df[df['label'] == 1]['title'].value_counts().head(8).to_dict()

    # Empirical Warning Indicators Matrix (Combining Structural Omissions, Contact Channels, and Linguistic Flags)
    combined_lower = combined_raw.str.lower()
    
    pay_pat = r'\b(no experience (required|needed)|earn \$[4-9]\d{1,2}|earn \$[1-9]\d{3}|no interview|make money fast|instant cash)\b'
    bank_pat = r'\b(social security|ssn|bank account details|routing number|credit card|date of birth)\b'
    urgent_pat = r'\b(urgent hiring|immediate placement|start today|urgent vacancy|apply immediately|limited spots)\b'
    
    has_pay = combined_lower.str.contains(pay_pat, regex=True)
    has_bank = combined_lower.str.contains(bank_pat, regex=True)
    has_urgent = combined_lower.str.contains(urgent_pat, regex=True)
    has_email = combined_raw.str.contains('#EMAIL_')
    has_phone = combined_raw.str.contains('#PHONE_')
    
    no_logo = ~df['has_company_logo'].astype(str).str.lower().isin(['t', 'true', '1'])
    no_profile = df['company_profile'].str.strip() == ''
    no_questions = ~df['has_questions'].astype(str).str.lower().isin(['t', 'true', '1'])
    is_remote = df['telecommuting'].astype(str).str.lower().isin(['t', 'true', '1'])

    indicator_catalog = [
        {
            'name': 'Unrealistic Pay / No Experience',
            'category': 'Linguistic Bait',
            'severity': 'HIGH',
            'legit_pct': round(float(has_pay[df['label'] == 0].mean() * 100), 2),
            'fraud_pct': round(float(has_pay[df['label'] == 1].mean() * 100), 2),
            'description': 'Mentions $40-$100+/hr with no experience or instant cash offers.'
        },
        {
            'name': 'Direct Email Token (#EMAIL_)',
            'category': 'Contact Redirection',
            'severity': 'HIGH',
            'legit_pct': round(float(has_email[df['label'] == 0].mean() * 100), 2),
            'fraud_pct': round(float(has_email[df['label'] == 1].mean() * 100), 2),
            'description': 'Directs applicant to external personal email addresses instead of official portal.'
        },
        {
            'name': 'Direct Phone Token (#PHONE_)',
            'category': 'Contact Redirection',
            'severity': 'MEDIUM',
            'legit_pct': round(float(has_phone[df['label'] == 0].mean() * 100), 2),
            'fraud_pct': round(float(has_phone[df['label'] == 1].mean() * 100), 2),
            'description': 'Solocits direct SMS/calling bypassing corporate ATS application tracks.'
        },
        {
            'name': 'Missing Company Profile',
            'category': 'Identity Omission',
            'severity': 'HIGH',
            'legit_pct': round(float(no_profile[df['label'] == 0].mean() * 100), 2),
            'fraud_pct': round(float(no_profile[df['label'] == 1].mean() * 100), 2),
            'description': 'Listing leaves company background blank to hide identity or legitimacy.'
        },
        {
            'name': 'Missing Company Logo',
            'category': 'Identity Omission',
            'severity': 'MEDIUM',
            'legit_pct': round(float(no_logo[df['label'] == 0].mean() * 100), 2),
            'fraud_pct': round(float(no_logo[df['label'] == 1].mean() * 100), 2),
            'description': 'Postings created anonymously without corporate branding assets.'
        },
        {
            'name': 'Premature Sensitive / Banking Ask',
            'category': 'Identity Theft',
            'severity': 'CRITICAL',
            'legit_pct': round(float(has_bank[df['label'] == 0].mean() * 100), 2),
            'fraud_pct': round(float(has_bank[df['label'] == 1].mean() * 100), 2),
            'description': 'Requests SSN, date of birth, or bank account routing numbers upfront.'
        },
        {
            'name': 'Absence of Screening Questions',
            'category': 'Due Diligence Lack',
            'severity': 'LOW',
            'legit_pct': round(float(no_questions[df['label'] == 0].mean() * 100), 2),
            'fraud_pct': round(float(no_questions[df['label'] == 1].mean() * 100), 2),
            'description': 'No qualifying or technical screening questions before accepting resumes.'
        },
        {
            'name': 'Telecommuting / Remote Work Claim',
            'category': 'Targeting Bait',
            'severity': 'MEDIUM',
            'legit_pct': round(float(is_remote[df['label'] == 0].mean() * 100), 2),
            'fraud_pct': round(float(is_remote[df['label'] == 1].mean() * 100), 2),
            'description': 'Advertises 100% remote work to maximize broad applicant volume.'
        }
    ]

    for item in indicator_catalog:
        l = item['legit_pct']
        f = item['fraud_pct']
        item['risk_multiplier'] = round(f / l, 2) if l > 0 else 0.0

    # Step 2: Extract Model Features & Feature Importance
    print("Step 2: Loading model artifacts for text feature extraction...")
    tfidf = joblib.load('models/tfidf_vectorizer.joblib')
    lr_balanced = joblib.load('models/lr_balanced.joblib')
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = lr_balanced.coef_[0]

    top_fraud_idx = np.argsort(coefs)[-30:][::-1]
    top_legit_idx = np.argsort(coefs)[:30]

    top_fraud_features = [
        {'term': feature_names[i], 'weight': round(float(coefs[i]), 4)}
        for i in top_fraud_idx
    ]
    top_legit_features = [
        {'term': feature_names[i], 'weight': round(float(abs(coefs[i])), 4), 'raw_weight': round(float(coefs[i]), 4)}
        for i in top_legit_idx
    ]

    # Step 3: Load Model Performance Metrics
    print("Step 3: Loading model comparison metrics...")
    metrics_df = pd.read_csv('model_comparison_results.csv', index_col=0)
    model_metrics = metrics_df.to_dict(orient='index')

    # Save all precomputed data into JSON
    insights = {
        'dataset': {
            'total_records': total_records,
            'fraud_count': fraud_count,
            'legit_count': legit_count,
            'fraud_pct': round(fraud_pct, 2),
            'legit_pct': round(legit_pct, 2),
            'metadata_flags': metadata_flags,
            'missing_rates': missing_rates,
            'text_lengths': text_lengths,
            'token_stats': token_stats,
            'top_fraud_industries': top_fraud_industries,
            'top_fraud_functions': top_fraud_functions,
            'top_fraud_titles': top_fraud_titles
        },
        'warning_indicators': indicator_catalog,
        'features': {
            'top_fraud_features': top_fraud_features,
            'top_legit_features': top_legit_features
        },
        'models': model_metrics
    }

    with open('models/dataset_insights.json', 'w', encoding='utf-8') as f:
        json.dump(insights, f, indent=2)
    print("Saved dataset insights to models/dataset_insights.json")

    # Step 4: Generate Publication-Quality Visualizations (PNG)
    print("Step 4: Generating visual charts...")
    
    # ------------------ Chart 1: Dataset Distribution ------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Dataset Distribution & Exploratory Data Analysis (EMScAD Job Postings)', fontsize=15, fontweight='bold', y=0.98)

    # 1.1 Class Imbalance (Donut Chart)
    ax = axes[0, 0]
    sizes = [legit_count, fraud_count]
    labels = [f"Legitimate\n{legit_count:,} ({legit_pct:.1f}%)", f"Fraudulent\n{fraud_count:,} ({fraud_pct:.1f}%)"]
    colors = ['#3B82F6', '#EF4444']
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=30,
        colors=colors, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
        pctdistance=0.75, textprops=dict(fontsize=10, fontweight='bold')
    )
    ax.set_title('Severe Class Imbalance (17,880 Total Postings)', fontsize=12, fontweight='bold')

    # 1.2 Metadata Indicators
    ax = axes[0, 1]
    keys = ['has_company_logo', 'has_questions', 'telecommuting']
    clean_keys = ['Company Logo Present', 'Screening Questions', 'Telecommuting / Remote']
    legit_vals = [metadata_flags[k]['legitimate'] for k in keys]
    fraud_vals = [metadata_flags[k]['fraudulent'] for k in keys]
    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w/2, legit_vals, width=w, label='Legitimate', color='#3B82F6', alpha=0.9)
    ax.bar(x + w/2, fraud_vals, width=w, label='Fraudulent', color='#EF4444', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_keys, fontsize=10)
    ax.set_ylabel('% of Postings', fontsize=10)
    ax.set_title('Metadata Attribute Comparison (% Present)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True)
    for i in range(len(keys)):
        ax.annotate(f"{legit_vals[i]}%", (x[i] - w/2, legit_vals[i] + 1.5), ha='center', fontsize=9)
        ax.annotate(f"{fraud_vals[i]}%", (x[i] + w/2, fraud_vals[i] + 1.5), ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 100)

    # 1.3 Missing Data Rate Comparison
    ax = axes[1, 0]
    m_keys = ['company_profile', 'requirements', 'benefits', 'salary_range']
    m_labels = ['Company Profile', 'Requirements', 'Benefits', 'Salary Range']
    legit_m = [missing_rates[k]['legitimate'] for k in m_keys]
    fraud_m = [missing_rates[k]['fraudulent'] for k in m_keys]
    x_m = np.arange(len(m_keys))
    ax.bar(x_m - w/2, legit_m, width=w, label='Legitimate Missing %', color='#60A5FA')
    ax.bar(x_m + w/2, fraud_m, width=w, label='Fraudulent Missing %', color='#F87171')
    ax.set_xticks(x_m)
    ax.set_xticklabels(m_labels, fontsize=10)
    ax.set_ylabel('% Missing in Postings', fontsize=10)
    ax.set_title('Field Omission / Missing Rates (% Blank)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True)
    for i in range(len(m_keys)):
        ax.annotate(f"{legit_m[i]}%", (x_m[i] - w/2, legit_m[i] + 1.5), ha='center', fontsize=9)
        ax.annotate(f"{fraud_m[i]}%", (x_m[i] + w/2, fraud_m[i] + 1.5), ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 100)

    # 1.4 Top Fraud Industries
    ax = axes[1, 1]
    ind_names = list(top_fraud_industries.keys())[:6]
    ind_counts = [top_fraud_industries[k] for k in ind_names]
    ax.barh(ind_names[::-1], ind_counts[::-1], color='#E11D48')
    ax.set_xlabel('Number of Fraudulent Postings', fontsize=10)
    ax.set_title('Top Targeted Industries for Fake Postings', fontsize=12, fontweight='bold')
    for idx, count in enumerate(ind_counts[::-1]):
        ax.annotate(f" {count}", (count, idx), va='center', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved dataset_distribution.png")

    # ------------------ Chart 2: Important Text Features ------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle('Key Text Features: Logistic Regression Model Coefficients', fontsize=15, fontweight='bold', y=0.98)

    # Top Fraud Terms
    ax = axes[0]
    f_terms = [item['term'] for item in top_fraud_features[:15]][::-1]
    f_weights = [item['weight'] for item in top_fraud_features[:15]][::-1]
    bars = ax.barh(f_terms, f_weights, color='#DC2626')
    ax.set_xlabel('Logistic Regression Weight (+ Fraud Indicator)', fontsize=11, fontweight='bold')
    ax.set_title('Top 15 Predictive Terms for FRAUDULENT Postings', fontsize=12, fontweight='bold', color='#991B1B')
    for bar in bars:
        w_val = bar.get_width()
        ax.annotate(f" {w_val:.2f}", (w_val, bar.get_y() + bar.get_height()/2), va='center', fontsize=9, fontweight='bold')

    # Top Legit Terms
    ax = axes[1]
    l_terms = [item['term'] for item in top_legit_features[:15]][::-1]
    l_weights = [item['weight'] for item in top_legit_features[:15]][::-1]
    bars_l = ax.barh(l_terms, l_weights, color='#059669')
    ax.set_xlabel('Logistic Regression Weight (+ Legitimate Indicator)', fontsize=11, fontweight='bold')
    ax.set_title('Top 15 Predictive Terms for LEGITIMATE Postings', fontsize=12, fontweight='bold', color='#065F46')
    for bar in bars_l:
        w_val = bar.get_width()
        ax.annotate(f" {w_val:.2f}", (w_val, bar.get_y() + bar.get_height()/2), va='center', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('text_features_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved text_features_importance.png")

    # ------------------ Chart 3: Model Performance Comparison ------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Model Evaluation & Performance Benchmarking', fontsize=15, fontweight='bold', y=0.98)

    # 3.1 Bar Chart of Key Metrics
    ax = axes[0]
    models = list(model_metrics.keys())
    metrics_to_show = ['Precision (Fraud)', 'Recall (Fraud)', 'F1-Score (Fraud)', 'ROC-AUC']
    x = np.arange(len(metrics_to_show))
    width = 0.25

    colors_models = ['#2563EB', '#D97706', '#059669']
    for idx, m_name in enumerate(models):
        vals = [model_metrics[m_name][metric] for metric in metrics_to_show]
        rects = ax.bar(x + idx * width, vals, width, label=m_name, color=colors_models[idx])
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.2f}', (rect.get_x() + rect.get_width()/2, h + 0.015),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_to_show, fontsize=10, fontweight='bold')
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=11)
    ax.set_title('Performance Metrics on Unseen Test Set (20% Stratified)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower left', frameon=True)

    # 3.2 Confusion Matrix Detail for Best Operational Model (LR Balanced)
    ax = axes[1]
    best_m = 'Logistic Regression (Balanced)'
    tn = int(model_metrics[best_m]['TN'])
    fp = int(model_metrics[best_m]['FP'])
    fn = int(model_metrics[best_m]['FN'])
    tp = int(model_metrics[best_m]['TP'])
    cm = np.array([[tn, fp], [fn, tp]])
    
    im = ax.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    ax.set_title(f'Confusion Matrix: {best_m}\n(Catches {tp}/{tp+fn} Scams = 88.4% Recall)', fontsize=12, fontweight='bold')
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Pred Legit', 'Pred Fraud'], fontsize=11)
    ax.set_yticklabels(['Actual Legit', 'Actual Fraud'], fontsize=11)

    thresh = cm.max() / 2.0
    labels_cm = [
        [f"True Negative (TN)\n{tn:,}\n(Legit accurately verified)", f"False Positive (FP)\n{fp:,}\n(Legit flagged for review)"],
        [f"False Negative (FN)\n{fn:,}\n(Scam missed - Critical Risk)", f"True Positive (TP)\n{tp:,}\n(Scam successfully caught!)"]
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels_cm[i][j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('model_performance_visualized.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved model_performance_visualized.png")

    # ------------------ Chart 4: Warning Indicators & Red Flags ------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.suptitle('Common Warning Indicators & Empirical Threat Patterns', fontsize=15, fontweight='bold', y=0.98)

    # 4.1 Indicator Prevalence (% in Legit vs % in Fraud)
    ax = axes[0]
    ind_labels = [item['name'] for item in indicator_catalog][::-1]
    f_pcts = [item['fraud_pct'] for item in indicator_catalog][::-1]
    l_pcts = [item['legit_pct'] for item in indicator_catalog][::-1]
    y_pos = np.arange(len(ind_labels))
    w = 0.35

    ax.barh(y_pos - w/2, l_pcts, height=w, label='Legitimate Postings', color='#3B82F6')
    ax.barh(y_pos + w/2, f_pcts, height=w, label='Fraudulent Postings', color='#EF4444')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ind_labels, fontsize=9.5)
    ax.set_xlabel('% Prevalence in Postings', fontsize=10, fontweight='bold')
    ax.set_title('Red Flag Pattern Frequency Comparison (% of Postings)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    for i in range(len(ind_labels)):
        ax.annotate(f"{l_pcts[i]}%", (l_pcts[i] + 0.8, y_pos[i] - w/2), va='center', fontsize=8.5)
        ax.annotate(f"{f_pcts[i]}%", (f_pcts[i] + 0.8, y_pos[i] + w/2), va='center', fontsize=8.5, fontweight='bold')
    ax.set_xlim(0, 85)

    # 4.2 Risk Multiplier Ratio
    ax = axes[1]
    multipliers = [item['risk_multiplier'] for item in indicator_catalog][::-1]
    bar_colors = ['#EF4444' if m >= 3.0 else ('#F59E0B' if m >= 1.5 else '#3B82F6') for m in multipliers]
    bars_m = ax.barh(ind_labels, multipliers, color=bar_colors)
    ax.set_xlabel('Risk Multiplier (Fraud % / Legit %)', fontsize=10, fontweight='bold')
    ax.set_title('Relative Scam Risk Multiplier per Indicator', fontsize=12, fontweight='bold')
    ax.axvline(1.0, color='#6B7280', linestyle='--', linewidth=1, label='Baseline (1.0x)')
    for bar in bars_m:
        w_val = bar.get_width()
        ax.annotate(f" {w_val:.1f}x", (w_val, bar.get_y() + bar.get_height()/2), va='center', fontsize=9, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('warning_indicators_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved warning_indicators_distribution.png")

    print("\nAll visualizations and precomputed summaries generated successfully!")

if __name__ == '__main__':
    analyze_and_generate()
