import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set page configuration
st.set_page_config(
    page_title="Job Fraud Shield | AI Scam Analytics & Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished dashboard UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    .risk-card {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        color: #991B1B;
    }
    .risk-medium {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        color: #92400E;
    }
    .risk-low {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        color: #065F46;
    }
    .badge-critical {
        background-color: #7F1D1D;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-high {
        background-color: #DC2626;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-medium {
        background-color: #D97706;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-low {
        background-color: #2563EB;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Cache resource loaders
@st.cache_resource
def load_nltk_resources():
    try:
        stops = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stops = set(stopwords.words('english'))
    try:
        lem = WordNetLemmatizer()
        lem.lemmatize('test')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        lem = WordNetLemmatizer()
    return stops, lem

stops, lemmatizer = load_nltk_resources()

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

@st.cache_resource
def load_models_and_artifacts():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    tfidf = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    lr_balanced = joblib.load(os.path.join(models_dir, 'lr_balanced.joblib'))
    lr_default = joblib.load(os.path.join(models_dir, 'lr_default.joblib'))
    nb_model = joblib.load(os.path.join(models_dir, 'naive_bayes.joblib'))
    
    samples_path = os.path.join(models_dir, 'sample_jobs.json')
    if os.path.exists(samples_path):
        with open(samples_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
    else:
        samples = {}
        
    insights_path = os.path.join(models_dir, 'dataset_insights.json')
    if os.path.exists(insights_path):
        with open(insights_path, 'r', encoding='utf-8') as f:
            insights = json.load(f)
    else:
        insights = {}

    return {
        'tfidf': tfidf,
        'Logistic Regression (Balanced)': lr_balanced,
        'Logistic Regression (Standard)': lr_default,
        'Multinomial Naive Bayes': nb_model,
        'samples': samples,
        'insights': insights
    }

from heuristics import scan_risk_indicators

def explain_prediction(model, tfidf, cleaned_text, top_k=10):
    if not hasattr(model, 'coef_'):
        return None
        
    feature_names = np.array(tfidf.get_feature_names_out())
    tfidf_vec = tfidf.transform([cleaned_text])
    nonzero_indices = tfidf_vec.nonzero()[1]
    
    if len(nonzero_indices) == 0:
        return None
        
    weights = model.coef_[0][nonzero_indices]
    values = tfidf_vec.data
    contributions = weights * values
    terms = feature_names[nonzero_indices]
    
    df_contrib = pd.DataFrame({
        'term': terms,
        'contribution': contributions,
        'weight': weights,
        'tfidf': values
    })
    
    fraud_indicators = df_contrib.sort_values(by='contribution', ascending=False).head(top_k)
    legit_indicators = df_contrib.sort_values(by='contribution', ascending=True).head(top_k)
    
    return {
        'fraud_leaning': fraud_indicators[fraud_indicators['contribution'] > 0],
        'legit_leaning': legit_indicators[legit_indicators['contribution'] < 0]
    }

# Load assets
artifacts = load_models_and_artifacts()
samples = artifacts['samples']
insights = artifacts['insights']

# App Header
st.markdown('<div class="main-header">🛡️ Job Fraud Shield</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comprehensive AI Detection & Analytics Platform: Dataset Distribution, Feature Attribution, Model Benchmarking & Warning Indicators</div>', unsafe_allow_html=True)

# Tabs Navigation
tab_detector, tab_dist, tab_features, tab_perf, tab_indicators = st.tabs([
    "🔍 Real-Time Scam Scanner",
    "📊 Dataset Distribution & EDA",
    "🔤 Important Text Features",
    "📈 Model Performance & Evaluation",
    "🚩 Common Warning Indicators"
])

# ==============================================================================
# TAB 1: REAL-TIME SCAM SCANNER
# ==============================================================================
with tab_detector:
    st.sidebar.header("⚙️ Scanner Settings")
    model_choice = st.sidebar.selectbox(
        "Classifier Model",
        options=[
            "Logistic Regression (Balanced)",
            "Logistic Regression (Standard)",
            "Multinomial Naive Bayes"
        ],
        index=0,
        help="Balanced Logistic Regression delivers 88.4% recall on fraudulent postings. Standard maximizes precision."
    )

    decision_threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Lower threshold to capture more suspected frauds (higher recall). Raise to only flag high-certainty scams."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Preloaded Examples")
    sample_choice = st.sidebar.selectbox(
        "Select benchmark job posting:",
        options=["Custom Input"] + list(samples.keys()),
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Quick Model Specs:**
    - **LR (Balanced):** Recall 88.4%, F1 0.77, AUC 0.984
    - **Naive Bayes:** Precision 85.1%, Recall 69.4%
    - **LR (Standard):** Precision 98.8%, Recall 45.7%
    """)

    default_text = ""
    if sample_choice != "Custom Input" and sample_choice in samples:
        default_text = samples[sample_choice]

    st.subheader("Job Listing Text Input")
    user_text = st.text_area(
        "Paste Job Posting Details (Title, Description, Requirements, Benefits, etc.):",
        value=default_text,
        height=220,
        placeholder="Paste full job posting content here to evaluate..."
    )

    col_btn, col_stats = st.columns([1, 4])
    with col_btn:
        analyze_btn = st.button("🔍 Run Fraud Analysis", type="primary", use_container_width=True)

    with col_stats:
        words = len(user_text.split())
        chars = len(user_text)
        st.caption(f"Input stats: **{words:,}** words | **{chars:,}** characters")

    # Run analysis
    if analyze_btn or (user_text.strip() != "" and sample_choice != "Custom Input"):
        if not user_text.strip():
            st.warning("Please paste or select a job posting to analyze.")
        else:
            cleaned = clean_input_text(user_text)
            
            if not cleaned:
                st.error("Input text does not contain sufficient alphabetic words after cleaning.")
            else:
                model = artifacts[model_choice]
                tfidf = artifacts['tfidf']
                
                vec = tfidf.transform([cleaned])
                prob_fraud = float(model.predict_proba(vec)[0][1])
                prob_legit = float(model.predict_proba(vec)[0][0])
                is_fraud = prob_fraud >= decision_threshold
                
                rule_flags = scan_risk_indicators(user_text)
                
                st.markdown("---")
                st.subheader("📊 Prediction & Risk Assessment")
                
                if is_fraud:
                    st.markdown(f"""
                    <div class="risk-card risk-high">
                        <h3 style="margin:0; font-size:1.4rem;">🚨 HIGH RISK / SUSPECTED FRAUDULENT POSTING</h3>
                        <p style="margin:5px 0 0 0; font-size:1.0rem;">
                            This job posting exhibits strong language patterns consistent with employment scams and phishing attempts.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prob_fraud >= 0.30:
                    st.markdown(f"""
                    <div class="risk-card risk-medium">
                        <h3 style="margin:0; font-size:1.4rem;">⚠️ MODERATE RISK / PROCEED WITH CAUTION</h3>
                        <p style="margin:5px 0 0 0; font-size:1.0rem;">
                            This listing contains ambiguous or mildly suspicious elements. Exercise caution and verify employer credentials.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-card risk-low">
                        <h3 style="margin:0; font-size:1.4rem;">✅ LIKELY LEGITIMATE POSTING</h3>
                        <p style="margin:5px 0 0 0; font-size:1.0rem;">
                            The language aligns with established legitimate hiring listings.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric(label="Fraud Probability", value=f"{prob_fraud * 100:.1f}%")
                with mcol2:
                    st.metric(label="Legitimate Probability", value=f"{prob_legit * 100:.1f}%")
                with mcol3:
                    risk_lvl = "High" if prob_fraud >= 0.60 else ("Medium" if prob_fraud >= 0.30 else "Low")
                    st.metric(label="Calculated Risk Level", value=risk_lvl)
                with mcol4:
                    confidence = max(prob_fraud, prob_legit) * 100
                    st.metric(label="Model Confidence", value=f"{confidence:.1f}%")

                st.markdown("**Scam Risk Meter:**")
                st.progress(float(prob_fraud))
                
                st.markdown("---")
                st.subheader("🚩 Detected Risk Indicators & Heuristic Red Flags")
                
                if rule_flags:
                    for flag in rule_flags:
                        badge_color = "🔴" if flag['severity'] == "HIGH" else ("🟠" if flag['severity'] == "MEDIUM" else "🟡")
                        with st.expander(f"{badge_color} **{flag['category']}** ({flag['severity']} Severity)", expanded=True):
                            st.write(f"**Trigger Detail:** {flag['detail']}")
                            st.write(f"**Identified Phrases:** `{', '.join(flag['matched'])}`")
                            st.info(f"💡 **Safety Tip:** {flag['tip']}")
                else:
                    st.success("✔️ No explicit heuristic red flags (e.g., wire transfer keywords, Telegram contact, check cashing) detected in the raw text.")

                st.markdown("---")
                st.subheader("🔍 Prediction Attribution (Feature Contribution)")
                
                explanation = explain_prediction(model, tfidf, cleaned, top_k=8)
                
                if explanation is not None:
                    exp_col1, exp_col2 = st.columns(2)
                    
                    with exp_col1:
                        st.markdown("##### 🚨 Top Fraud-Indicative Terms in This Text")
                        fraud_df = explanation['fraud_leaning']
                        if not fraud_df.empty:
                            fig, ax = plt.subplots(figsize=(6, 3.2))
                            sorted_f = fraud_df.sort_values(by='contribution', ascending=True)
                            ax.barh(sorted_f['term'], sorted_f['contribution'], color='#EF4444')
                            ax.set_xlabel('Contribution Score (+ Fraud)', fontsize=9)
                            ax.set_title('Tokens Driving Fraud Score', fontsize=10, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.write("No strong fraud-associated tokens identified.")

                    with exp_col2:
                        st.markdown("##### 🏢 Top Legitimate-Indicative Terms in This Text")
                        legit_df = explanation['legit_leaning']
                        if not legit_df.empty:
                            fig, ax = plt.subplots(figsize=(6, 3.2))
                            sorted_l = legit_df.sort_values(by='contribution', ascending=False)
                            ax.barh(sorted_l['term'], -sorted_l['contribution'], color='#10B981')
                            ax.set_xlabel('Contribution Score (+ Legit)', fontsize=9)
                            ax.set_title('Tokens Driving Legitimate Score', fontsize=10, fontweight='bold')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.write("No strong legitimate-associated tokens identified.")
                else:
                    st.info("Feature weight attribution is directly computed for linear models (Logistic Regression). For Naive Bayes, classification is based on joint word likelihood probabilities.")

# ==============================================================================
# TAB 2: DATASET DISTRIBUTION & EDA
# ==============================================================================
with tab_dist:
    st.subheader("📊 Dataset Distribution & Exploratory Data Analysis")
    st.write("""
    Analysis of the **EMScAD (Employment Scam Aegean Dataset)** containing **17,880** real-world job postings.
    Reveals extreme class imbalance, missing metadata patterns, and targeted industry sectors.
    """)

    # Top KPI Metrics Row
    d_info = insights.get('dataset', {})
    total_rec = d_info.get('total_records', 17880)
    legit_cnt = d_info.get('legit_count', 17014)
    fraud_cnt = d_info.get('fraud_count', 866)
    fraud_pct = d_info.get('fraud_pct', 4.84)
    legit_pct = d_info.get('legit_pct', 95.16)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Job Postings</div>
            <div class="kpi-val">{total_rec:,}</div>
            <div class="kpi-sub">Curated Global Dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Legitimate Postings</div>
            <div class="kpi-val" style="color:#2563EB;">{legit_cnt:,}</div>
            <div class="kpi-sub">{legit_pct}% of total data</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Fraudulent Postings</div>
            <div class="kpi-val" style="color:#EF4444;">{fraud_cnt:,}</div>
            <div class="kpi-sub">{fraud_pct}% of total data</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        imbalance_ratio = round(legit_cnt / fraud_cnt, 1)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Class Imbalance Ratio</div>
            <div class="kpi-val" style="color:#D97706;">{imbalance_ratio} : 1</div>
            <div class="kpi-sub">1 Scam per 20 Postings</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Display Composite Distribution Figure
    if os.path.exists('dataset_distribution.png'):
        st.image('dataset_distribution.png', caption='Figure 1: Dataset Distribution & Metadata Comparative Overview', use_container_width=True)

    # Detailed Interactive Drill-down
    st.markdown("---")
    st.subheader("🔍 Key Structural Discrepancies (Fraud vs Legitimate)")

    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        st.markdown("#### 1. Identity & Metadata Signals")
        meta_data = d_info.get('metadata_flags', {})
        df_meta = pd.DataFrame([
            {
                'Attribute': 'Company Logo Present',
                'Legitimate (%)': f"{meta_data.get('has_company_logo', {}).get('legitimate', 81.91)}%",
                'Fraudulent (%)': f"{meta_data.get('has_company_logo', {}).get('fraudulent', 32.68)}%",
                'Insight': 'Fake postings are 2.5x less likely to include a corporate logo.'
            },
            {
                'Attribute': 'Screening Questions',
                'Legitimate (%)': f"{meta_data.get('has_questions', {}).get('legitimate', 50.21)}%",
                'Fraudulent (%)': f"{meta_data.get('has_questions', {}).get('fraudulent', 28.87)}%",
                'Insight': 'Scammers avoid screening questions to lower applicant friction.'
            },
            {
                'Attribute': 'Telecommuting / Remote',
                'Legitimate (%)': f"{meta_data.get('telecommuting', {}).get('legitimate', 4.13)}%",
                'Fraudulent (%)': f"{meta_data.get('telecommuting', {}).get('fraudulent', 7.39)}%",
                'Insight': 'Scammers offer remote work nearly 2x as often to lure victims nationwide.'
            }
        ])
        st.dataframe(df_meta, use_container_width=True, hide_index=True)

        st.markdown("#### 2. Average Word Count per Section")
        t_lens = d_info.get('text_lengths', {})
        df_lens = pd.DataFrame([
            {
                'Field': 'Company Profile',
                'Legit Mean Words': t_lens.get('company_profile', {}).get('legit_avg_words', 99.1),
                'Fraud Mean Words': t_lens.get('company_profile', {}).get('fraud_avg_words', 34.3),
                'Difference': 'Fraud is 65% shorter (or completely blank)'
            },
            {
                'Field': 'Requirements',
                'Legit Mean Words': t_lens.get('requirements', {}).get('legit_avg_words', 91.1),
                'Fraud Mean Words': t_lens.get('requirements', {}).get('fraud_avg_words', 63.2),
                'Difference': 'Scam job requirements are vague and brief'
            },
            {
                'Field': 'Description',
                'Legit Mean Words': t_lens.get('description', {}).get('legit_avg_words', 186.9),
                'Fraud Mean Words': t_lens.get('description', {}).get('fraud_avg_words', 167.6),
                'Difference': 'Descriptions are similarly sized but use different vocabulary'
            }
        ])
        st.dataframe(df_lens, use_container_width=True, hide_index=True)

    with dist_col2:
        st.markdown("#### 3. Field Omission / Missing Value Rates")
        miss = d_info.get('missing_rates', {})
        df_miss = pd.DataFrame([
            {
                'Field': 'Company Profile',
                'Legitimate Missing': f"{miss.get('company_profile', {}).get('legitimate', 15.99)}%",
                'Fraudulent Missing': f"{miss.get('company_profile', {}).get('fraudulent', 67.78)}%",
                'Risk Ratio': '4.2x higher missing rate in scams'
            },
            {
                'Field': 'Salary Range',
                'Legitimate Missing': f"{miss.get('salary_range', {}).get('legitimate', 84.45)}%",
                'Fraudulent Missing': f"{miss.get('salary_range', {}).get('fraudulent', 74.25)}%",
                'Risk Ratio': 'Both high; scams often state fake high pay in text'
            },
            {
                'Field': 'Department',
                'Legitimate Missing': f"{miss.get('department', {}).get('legitimate', 64.75)}%",
                'Fraudulent Missing': f"{miss.get('department', {}).get('fraudulent', 61.32)}%",
                'Risk Ratio': 'Similar missing rate across both classes'
            }
        ])
        st.dataframe(df_miss, use_container_width=True, hide_index=True)

        st.markdown("#### 4. Top Targeted Fraud Industries")
        top_ind = d_info.get('top_fraud_industries', {})
        if top_ind:
            df_ind = pd.DataFrame(list(top_ind.items()), columns=['Industry', 'Fake Job Count'])
            st.dataframe(df_ind, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: IMPORTANT TEXT FEATURES
# ==============================================================================
with tab_features:
    st.subheader("🔤 Important Text Features & Vocabulary Attribution")
    st.write("""
    Features extracted using **TF-IDF Vectorization** (10,000 unigrams & bigrams) and ranked by their **Logistic Regression coefficients**.
    Positive coefficients directly predict fraudulent listings, while negative coefficients predict legitimate corporate listings.
    """)

    if os.path.exists('text_features_importance.png'):
        st.image('text_features_importance.png', caption='Figure 2: Top Model Feature Coefficients for Fraud vs Legitimate Postings', use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Interactive Feature Explorer")

    feat_data = insights.get('features', {})
    top_f = feat_data.get('top_fraud_features', [])
    top_l = feat_data.get('top_legit_features', [])

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("#### 🚨 Top Fraud-Predictive Keywords")
        st.caption("Higher coefficient = Higher probability that a listing containing this term is a scam.")
        df_top_f = pd.DataFrame(top_f[:20])
        df_top_f.columns = ['N-Gram Term', 'Model Weight (+ Fraud)']
        st.dataframe(df_top_f, use_container_width=True, hide_index=True)

    with fc2:
        st.markdown("#### ✅ Top Legitimate-Predictive Keywords")
        st.caption("Higher magnitude = Higher probability that a listing containing this term is authentic.")
        df_top_l = pd.DataFrame(top_l[:20])
        df_top_l = df_top_l[['term', 'weight']]
        df_top_l.columns = ['N-Gram Term', 'Model Weight (+ Legit)']
        st.dataframe(df_top_l, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("💡 Linguistic Insights: Why These Features Matter")
    st.markdown("""
    - **`link`, `phone`, `email`**: Scammers actively push applicants to click external phishing links or contact external webmail/phone addresses.
    - **`data entry`, `clerk`, `receptionist`, `assistant`**: Classic entry-level titles used as bait because they require minimal technical prerequisites and attract huge applicant volumes.
    - **`work home`, `earn`, `money`, `cash`, `signing bonus`**: Financial enticements that distract candidates from doing due diligence.
    - **`subsea`, `offshore`, `oil gas`**: A known scam campaign in the dataset impersonating oil & gas conglomerates with lucrative overseas contracts.
    - **`client`, `team`, `digital`, `startup`, `software`, `recruitment`**: Genuine vocabulary found in authentic corporate job postings detailing team collaboration and technology stacks.
    """)

# ==============================================================================
# TAB 4: MODEL PERFORMANCE & EVALUATION
# ==============================================================================
with tab_perf:
    st.subheader("📈 Model Evaluation & Performance Benchmarking")
    st.write("""
    Evaluation of three candidate architectures on a stratified **20% held-out test set (3,576 unseen postings)**.
    In fraud detection, **Recall** is prioritized over raw accuracy because a missed scam (False Negative) can result in severe financial harm to job seekers.
    """)

    if os.path.exists('model_performance_visualized.png'):
        st.image('model_performance_visualized.png', caption='Figure 3: Benchmark Metrics Comparison & Confusion Matrix for Balanced Logistic Regression', use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Comprehensive Test-Set Benchmark Table")

    models_info = insights.get('models', {})
    if models_info:
        table_rows = []
        for m_name, m_stats in models_info.items():
            table_rows.append({
                'Model': m_name,
                'Accuracy': f"{m_stats.get('Accuracy', 0):.4f}",
                'Fraud Recall': f"{m_stats.get('Recall (Fraud)', 0)*100:.1f}%",
                'Fraud Precision': f"{m_stats.get('Precision (Fraud)', 0)*100:.1f}%",
                'Fraud F1': f"{m_stats.get('F1-Score (Fraud)', 0):.4f}",
                'ROC-AUC': f"{m_stats.get('ROC-AUC', 0):.4f}",
                'Caught Scams (TP)': int(m_stats.get('TP', 0)),
                'Missed Scams (FN)': int(m_stats.get('FN', 0)),
                'False Alarms (FP)': int(m_stats.get('FP', 0))
            })
        df_models = pd.DataFrame(table_rows)
        st.dataframe(df_models, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("⚖️ Trade-off Analysis: Why Balanced Logistic Regression is the Selected Model")

    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        st.markdown("""
        **1. Logistic Regression (Balanced) [CHOSEN]**
        - **Fraud Recall: 88.4%**
        - **ROC-AUC: 0.9842**
        - **Caught: 153 / 173 Scams**
        - Adjusts weights inversely proportional to class frequencies. Crucial for fraud security, minimizing critical false negatives.
        """)
    with tcol2:
        st.markdown("""
        **2. Multinomial Naive Bayes**
        - **Fraud Precision: 85.1%**
        - **Fraud Recall: 69.4%**
        - **Caught: 120 / 173 Scams**
        - Fast and lightweight probability estimates, but misses 53 fraudulent listings (30.6% false negative rate).
        """)
    with tcol3:
        st.markdown("""
        **3. Logistic Regression (Standard)**
        - **Fraud Precision: 98.8%**
        - **Fraud Recall: 45.7%**
        - **Missed: 94 / 173 Scams**
        - Suffers severely from class imbalance. While accurate when it flags, it fails to detect over half of all scams.
        """)

# ==============================================================================
# TAB 5: COMMON WARNING INDICATORS
# ==============================================================================
with tab_indicators:
    st.subheader("🚩 Common Warning Indicators & Threat Intelligence")
    st.write("""
    Empirical threat analysis combining **pattern frequency across the 17,880 dataset listings** and heuristic red flags used in modern employment scam syndicates.
    """)

    if os.path.exists('warning_indicators_distribution.png'):
        st.image('warning_indicators_distribution.png', caption='Figure 4: Red Flag Frequency Comparison and Relative Risk Multipliers', use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Empirical Warning Indicator Matrix")

    indicators_list = insights.get('warning_indicators', [])
    if indicators_list:
        ind_table = []
        for item in indicators_list:
            ind_table.append({
                'Indicator': item['name'],
                'Category': item['category'],
                'Severity': item['severity'],
                'Legitimate (%)': f"{item['legit_pct']}%",
                'Fraudulent (%)': f"{item['fraud_pct']}%",
                'Risk Multiplier': f"{item['risk_multiplier']}x",
                'Threat Description': item['description']
            })
        df_indicators = pd.DataFrame(ind_table)
        st.dataframe(df_indicators, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🛡️ Deep Dive: The 5 Most Prevalent Scam Typologies")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("""
        #### 1. The Fake Check / Advance-Fee Equipment Scam
        - **Pattern:** Scammer hires candidate immediately and sends a counterfeit cashier's check to "buy home office equipment" from a specified vendor.
        - **Threat:** When the victim wires money to the "vendor" (scammer), the check bounces days later, leaving the victim liable for thousands of dollars.
        - **Red Flags:** Mention of *wire transfer, cashier check, deposit check, Western Union, purchase equipment*.

        #### 2. The Unrealistic Pay / No-Experience Bait
        - **Pattern:** Offering $40-$80/hr or $200+/day for entry-level tasks like data entry, typist, or package inspection with zero interview or experience required.
        - **Threat:** Targets vulnerable job seekers, students, and remote workers looking for quick income.
        - **Red Flags:** *No experience needed, instant hiring, make money fast, flexible 2 hours daily*.
        """)

    with s2:
        st.markdown("""
        #### 3. Off-Platform Communication Redirection
        - **Pattern:** The listing instructs the applicant to message a private recruiter on Telegram, WhatsApp, Google Hangouts, or a free Gmail/Yahoo address.
        - **Threat:** Evades job board security moderation and anti-spam filters, preventing paper trails.
        - **Red Flags:** *text me on Telegram @..., send resume to hr-jobs@gmail.com, interview via WhatsApp*.

        #### 4. Premature Sensitive Data Solicitation (Identity Theft)
        - **Pattern:** Requesting Social Security Numbers (SSN), banking direct-deposit routing information, or credit card verification before an interview or formal offer.
        - **Threat:** Used immediately for financial fraud, synthetic identity creation, and unauthorized credit lines.
        - **Red Flags:** *Provide SSN for background check upfront, banking details for payroll portal setup*.
        """)

    st.markdown("---")
    st.subheader("✅ Actionable Candidate Safety Checklist")
    st.markdown("""
    1. **Domain Verification:** Always inspect email sender domains. Official recruiters use `@companyname.com`, never `@gmail.com` or spoofed domains (e.g., `@company-careers-portal.com`).
    2. **Refuse Upfront Financial Transactions:** Legitimate employers supply required hardware directly and never instruct employees to wire funds, purchase crypto, or deposit personal checks.
    3. **Official Job Board Cross-Referencing:** Always search for the requisition on the company's official corporate careers page or verified LinkedIn page.
    4. **Formal Screening Process:** Beware of offers extended without a two-way video interview or formal interview stages.
    """)
