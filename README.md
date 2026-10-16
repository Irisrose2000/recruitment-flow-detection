# 🛡️ Job Fraud Shield: AI Scam & Fraudulent Job Detection

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org)
[![NLP NLTK](https://img.shields.io/badge/NLP-NLTK%20TF--IDF-green.svg)](https://www.nltk.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, interpretable Machine Learning and Natural Language Processing (NLP) system designed to protect job seekers by identifying scam, phishing, and fraudulent employment advertisements. Powered by **TF-IDF N-Gram Vectorization**, a **Balanced Logistic Regression** classifier achieving **88.4% Recall** and **0.984 ROC-AUC**, a real-time **Heuristic Rule Engine**, and an interactive 5-tab **Streamlit** dashboard.

---

## 📌 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [Visual Dashboard Tour](#-visual-dashboard-tour)
- [Dataset Distribution & EDA](#-dataset-distribution--eda)
- [Important Text Features & Lexical Attribution](#-important-text-features--lexical-attribution)
- [Model Evaluation & Benchmarks](#-model-evaluation--benchmarks)
- [Common Warning Indicators & Threat Intelligence](#-common-warning-indicators--threat-intelligence)
- [Automated Testing Suite (Legitimate, Suspicious, Ambiguous)](#-automated-testing-suite)
- [System Limitations & Real-World Edge Cases](#-system-limitations--real-world-edge-cases)
- [Quickstart & Installation](#-quickstart--installation)
- [Project Directory Structure](#-project-directory-structure)
- [License](#-license)

---

## 🎯 Executive Overview

Employment scams have surged dramatically with the rise of remote hiring, inflicting millions of dollars in losses on job seekers through fake cashier check cashing, advance-fee equipment purchases, and identity theft. Detecting fake listings is an inherently asymmetric machine learning challenge due to **severe class imbalance**: fraudulent postings comprise under **5%** of all job listings. Standard classifiers optimized for raw accuracy fail to detect the majority of scams.

**Job Fraud Shield** addresses this challenge with a defense-in-depth architecture:
1. **Cost-Sensitive Machine Learning:** Uses Balanced Logistic Regression to heavily penalize False Negatives, catching **88.4% of scams** on unseen test data.
2. **Deterministic Heuristic Rules:** Instantly flags known threat patterns (wire transfer laundering, fake checks, Telegram/WhatsApp recruiting, premature SSN/banking demands).
3. **LIME-Inspired Token Attribution:** Deconstructs every prediction into specific word-level contribution scores, ensuring total transparency.
4. **Interactive 5-Tab Dashboard:** Allows recruiters, candidates, and data scientists to test postings, inspect distributions, review feature weights, and study threat intelligence.

---

## ⚡ Key Features

- **🔍 Real-Time Job Listing Scanner:** Paste any job posting or select from 8 pre-loaded benchmark cases to get an instant fraud probability, risk tier, and token attribution.
- **📊 Dataset Distribution & EDA Tab:** Visualizes the 17,880-row EMScAD dataset, showing class imbalance (19.6:1), metadata differences (logos, remote claims), and missing field rates.
- **🔤 Important Text Features Explorer:** Analyzes top predictive unigrams and bigrams for fraudulent vs. legitimate listings with model weight rankings.
- **📈 Comprehensive Model Benchmarking:** Compares Balanced Logistic Regression, Multinomial Naive Bayes, and Standard Logistic Regression with side-by-side metrics and confusion matrices.
- **🚩 Warning Indicators & Threat Matrix:** Catalogues empirical prevalence rates and risk multipliers across structural and linguistic red flags.
- **🧪 Automated Test Pipeline (`test_pipeline.py`):** Fully reproducible automated test suite evaluating legitimate, suspicious, and ambiguous test cases.

---

## 🖼️ Visual Dashboard Tour

The interactive Streamlit interface ([app.py](app.py)) provides five modular analytical views:

### 1. Real-Time Scam Scanner & Attribution
Evaluates raw text, computes multi-model consensus, fires heuristic red flags, and plots token-level attribution bars.
![Real-Time Scanner UI](screenshots/realtime_scanner_analysis.png)

### 2. Dataset Distribution & Exploratory Data Analysis
Displays the class imbalance donut chart, metadata flag discrepancies, missing rate comparisons, and top targeted fraud industries.
![Dataset Distribution](screenshots/dataset_distribution_eda.png)

### 3. Important Text Features & Vocabulary Importance
Examines the top 15 fraud-predictive keywords vs. top 15 legitimate-predictive keywords derived from model coefficients.
![Important Text Features](screenshots/important_text_features.png)

### 4. Model Performance & Evaluation Benchmarks
Side-by-side comparison of Precision, Recall, F1, and ROC-AUC on unseen test data, alongside the Confusion Matrix for Balanced LR.
![Model Performance](screenshots/model_performance_benchmarks.png)

### 5. Common Warning Indicators & Threat Intelligence
Empirical breakdown of red flag pattern frequency and relative risk multipliers across the 17,880 listings.
![Warning Indicators](screenshots/common_warning_indicators.png)

---

## 📊 Dataset Distribution & EDA

Trained and evaluated on the **EMScAD (Employment Scam Aegean Dataset)** containing **17,880** real-world job postings:

| Metric / Attribute | Legitimate Postings | Fraudulent Postings | Key Finding |
| :--- | :--- | :--- | :--- |
| **Total Postings** | 17,014 (95.16%) | 866 (4.84%) | **19.6 : 1 Imbalance** (1 scam per 20 jobs) |
| **Company Logo Present** | **81.91%** | **32.68%** | Scams are **2.5x less likely** to have a logo |
| **Screening Questions** | **50.21%** | **28.87%** | Scammers avoid screening questionnaires |
| **Telecommuting (Remote)** | **4.13%** | **7.39%** | Remote work claimed **nearly 2x as often** as bait |
| **Missing Company Profile** | **15.99%** | **67.78%** | Fraudulent listings are **4.2x more likely** to leave company bio blank |
| **Avg Profile Word Count** | **99.1 words** | **34.3 words** | Scam company profiles are 65% shorter or omitted |
| **Avg Requirements Word Count**| **91.1 words** | **63.2 words** | Scam job prerequisites are intentionally vague |

### Top Targeted Industries for Scams
1. **Oil & Energy** (109 listings) — Impersonating lucrative offshore engineering contracts.
2. **Accounting** (57 listings) — Fake payroll clerk and invoice processing schemes.
3. **Hospital & Health Care** (51 listings) — Remote medical data entry / billing typists.
4. **Marketing & Advertising** (45 listings) — Direct marketing / lead generation roles.
5. **Financial Services** (35 listings) — Wire transfer and payment disbursement schemes.

---

## 🔤 Important Text Features & Lexical Attribution

Using TF-IDF n-gram vectorization (10,000 features, unigrams + bigrams), the model coefficients ($\beta$) directly quantify feature importance:

### Top Fraud-Predictive Keywords ($+\beta$)
- `link` (+3.92): Directing candidates to external phishing forms or off-platform domains.
- `data entry` (+3.46): Most frequent entry-level bait title requiring minimal qualifications.
- `aptitude staffing` (+3.38): Recurrent fraudulent staffing entity identified in the dataset.
- `earn` (+3.31) / `money` (+3.14) / `cash` (+2.94): Focus on immediate monetary payout over job scope.
- `clerk` (+3.14) / `receptionist` (+2.77) / `administrative assistant` (+2.44): High-volume target titles.
- `signing bonus` (+2.90): Financial incentive used to distract candidates from verification.
- `work home` (+2.73): Remote work lure targeting stay-at-home parents and students.
- `subsea` (+3.08) / `oil gas` (+2.44) / `offshore` (+2.78): Replicated industry scam cluster.

### Top Legitimate-Predictive Keywords ($-\beta$)
- `client` (-3.53), `team` (-3.18), `digital` (-2.72), `english` (-2.67), `growing` (-2.44), `startup` (-2.39), `recruitment` (-2.30), `software` (-2.28), `web` (-2.13), `marketing` (-2.07).
- Reflect authentic corporate structure, team collaboration, technical skillsets, and customer deliverables.

---

## 📈 Model Evaluation & Benchmarks

Models were evaluated on a held-out **20% stratified test set** (3,576 unseen postings: 3,403 legitimate, 173 fraudulent):

| Classifier Model | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 | Macro F1 | ROC-AUC | Caught (TP) | Missed (FN) | False Alarms (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced) [CHOSEN]** | **97.40%** | 67.70% | **88.44%** | **0.7669** | **0.8766** | **0.9842** | **153** | **20** | 73 |
| **Multinomial Naive Bayes** | 97.93% | 85.11% | 69.36% | 0.7643 | 0.8768 | 0.9808 | 120 | 53 | 21 |
| **Logistic Regression (Standard)** | 97.34% | **98.75%** | 45.66% | 0.6245 | 0.8054 | 0.9783 | 79 | 94 | **1** |

### Why Recall Trumps Raw Accuracy in Fraud Detection
- In employment scam detection, **a False Negative (FN)** means a victim falls prey to check fraud or identity theft.
- **A False Positive (FP)** merely triggers an extra verification check (e.g., confirming the recruiter's corporate email).
- **Balanced Logistic Regression** catches **153 out of 173 scams** on the unseen test set, whereas standard Logistic Regression misses **94 scams (54% failure rate)** due to majority-class bias.

---

## 🚩 Common Warning Indicators & Threat Intelligence

| Indicator Name | Category | Severity | Legit (%) | Fraud (%) | Risk Multiplier | Threat Mechanism |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Unrealistic Pay / No Experience** | Linguistic Bait | **HIGH** | 0.24% | 3.58% | **15.2x** | Promises $40-$100+/hr with no experience required |
| **Missing Company Profile** | Identity Omission | **HIGH** | 15.99% | 67.78% | **4.2x** | Empty background hides syndicate identity |
| **Premature Sensitive Info** | Identity Theft | **CRITICAL** | 0.51% | 1.96% | **3.8x** | Solicits SSN, banking info, or ID cards prior to an interview |
| **Missing Company Logo** | Identity Omission | **MEDIUM** | 18.09% | 67.32% | **3.7x** | Postings submitted without verified branding |
| **Direct Email Token (`#EMAIL_`)** | Contact Redirection | **HIGH** | 7.07% | 20.09% | **2.8x** | Bypasses corporate ATS tracks to free webmail (Gmail/Yahoo) |
| **Direct Phone Token (`#PHONE_`)** | Contact Redirection | **MEDIUM** | 3.39% | 9.24% | **2.7x** | Directs applicant to SMS / WhatsApp / Telegram |
| **Telecommuting Claim** | Targeting Bait | **MEDIUM** | 4.13% | 7.39% | **1.8x** | Advertises remote flexibility to widen applicant capture |
| **Absence of Screening Questions** | Due Diligence Lack | **LOW** | 49.79% | 71.13% | **1.4x** | Zero qualification hurdles to maximize applicant volume |

---

## 🧪 Automated Testing Suite

The repository includes a dedicated automated test suite ([test_pipeline.py](test_pipeline.py)) validating **8 diverse benchmark cases**:

```bash
python test_pipeline.py
```

### Execution Matrix & Validation Results:
```text
=====================================================================================
JOB FRAUD SHIELD: AUTOMATED TEST SUITE (8 Benchmark Cases)
=====================================================================================
[PASSED] LEGITIMATE: Senior Cloud DevOps Engineer (Python/AWS)
         LR Balanced: 22.4% | Naive Bayes: 0.9% | LR Standard: 4.8%
         Flags: 0 -> [] | Verdict: LIKELY LEGITIMATE (LOW RISK)

[PASSED] LEGITIMATE: Enterprise Customer Success Manager (SaaS)
         LR Balanced: 7.5%  | Naive Bayes: 0.4% | LR Standard: 1.6%
         Flags: 0 -> [] | Verdict: LIKELY LEGITIMATE (LOW RISK)

[PASSED] SUSPICIOUS: Fake Check & Wire Transfer Clerk (Advance Fee)
         LR Balanced: 90.5% | Naive Bayes: 70.5% | LR Standard: 41.3%
         Flags: 4 -> [Financial, Unrealistic Pay, Off-Platform, Urgency]
         Verdict: SUSPECTED SCAM (HIGH RISK)

[PASSED] SUSPICIOUS: Telegram Crypto Investment Assistant (Immediate Cash)
         LR Balanced: 85.5% | Naive Bayes: 35.1% | LR Standard: 35.6%
         Flags: 4 -> [Financial, Unrealistic Pay, Off-Platform, Urgency]
         Verdict: SUSPECTED SCAM (HIGH RISK)

[PASSED] SUSPICIOUS: Identity Harvesting / Upfront SSN & Bank Details
         LR Balanced: 54.5% | Naive Bayes: 26.8% | LR Standard: 14.5%
         Flags: 3 -> [Off-Platform, Urgency, Sensitive Info Request]
         Verdict: SUSPECTED SCAM (HIGH RISK)

[PASSED] AMBIGUOUS: Stealth AI Startup ML Intern (Personal Gmail)
         LR Balanced: 10.5% | Naive Bayes: 0.1% | LR Standard: 2.5%
         Flags: 1 -> [Off-Platform Contact]
         Verdict: SUSPECTED SCAM (HIGH RISK due to personal Gmail address)

[PASSED] AMBIGUOUS: 100% Commission-Only Direct Sales (High Earning Claims)
         LR Balanced: 19.4% | Naive Bayes: 11.2% | LR Standard: 4.4%
         Flags: 1 -> [Unrealistic Compensation / No Experience]
         Verdict: PROCEED WITH CAUTION (MODERATE RISK)

[PASSED] AMBIGUOUS: Urgent Seasonal Warehouse Material Handler (Immediate Start)
         LR Balanced: 17.0% | Naive Bayes: 6.2% | LR Standard: 4.3%
         Flags: 1 -> [Urgent Placement Pressure]
         Verdict: PROCEED WITH CAUTION (MODERATE RISK)
=====================================================================================
Final Status: ALL TESTS PASSED SUCCESSFULLY (8/8)
```

---

## ⚠️ System Limitations & Real-World Edge Cases

While Job Fraud Shield delivers high recall and interpretability, deploying text-based scam detectors in production presents specific edge cases and limitations:

1. **Healthcare / Clinical Lexical Bias:**
   - *Observation:* Terms such as `hospital`, `medical`, and `nursing` carry slightly elevated fraud coefficients because the training data contains a cluster of 51 fake medical data entry listings.
   - *Mitigation:* While Balanced LR may assign ~50% probability to genuine clinical jobs, the Heuristic Engine reports **zero flags**, and Naive Bayes/Standard LR assign low probabilities (10-26%). The multi-model view prevents false alarms.

2. **Stealth-Mode Startups & Early-Stage Ventures:**
   - Legitimate seed-stage startups frequently lack corporate logos, official ATS career portals, or established domains (using personal Gmail or Substack).
   - Our system appropriately flags personal Gmail usage as a risk indicator. While this creates a minor inconvenience for stealth founders, it correctly prompts applicants to independently verify the founder's identity on LinkedIn or GitHub before submitting sensitive info.

3. **Vocabulary Drift & Synthetic LLM Scams:**
   - Sophisticated syndicates increasingly use large language models (ChatGPT, Claude) to draft grammatically flawless job postings that avoid colloquial scam phrasing (`make money fast`, `earn daily`).
   - *Solution:* The system integrates structural omissions (missing logo, blank profile, telecommuting mismatch) and regex pattern matching that trigger regardless of prose polish.

4. **Short Text & Token Sparsity:**
   - Single-sentence descriptions (e.g., *"Hiring drivers, call 555-0192"*) lack sufficient n-gram tokens for TF-IDF vectorization. In such cases, rule-based heuristics take precedence.

5. **Multilingual Constraints:**
   - The current NLTK lemmatization and TF-IDF pipeline is optimized for English-language postings. Non-English postings must be translated prior to evaluation.

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/job-fraud-shield.git
cd job-fraud-shield
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Deployment Modes (Ready for Production Action)

#### Mode A: Interactive Streamlit Dashboard (Port 8501)
```bash
streamlit run app.py
```
Access the 5-tab web dashboard at `http://localhost:8501`.

#### Mode B: Production REST API Microservice (Port 8000)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
- Interactive Swagger UI documentation: `http://localhost:8000/docs`
- Health probe: `GET /health`
- Real-time scoring: `POST /predict`
- High-throughput batch scoring: `POST /batch-predict`

#### Mode C: Command-Line Interface (CLI)
```bash
# Analyze a direct string
python cli.py "URGENT hiring data entry clerk earn $55/hr wire transfer money via western union" --explain

# Analyze from file
python cli.py --file path/to/posting.txt

# Pipe structured JSON to downstream tools
python cli.py --sample "Senior Cloud DevOps" --json
```

#### Mode D: Docker & Containerized Deployment
```bash
# Build and run with Docker
docker build -t job-fraud-shield .
docker run -p 8501:8501 job-fraud-shield

# Or run both Streamlit UI (8501) and FastAPI API (8000) via Docker Compose
docker-compose up --build
```

#### Mode E: Python Embedded Library API
```python
from model import JobFraudModelPipeline

pipeline = JobFraudModelPipeline()
pipeline.load("models")

# Real-time inference
result = pipeline.predict("Job description text here...")
print(result)

# Feature attribution
explanation = pipeline.explain("Job description text here...")
```

### 5. Run Automated Tests
```bash
python test_pipeline.py
```

---

## 📁 Project Directory Structure

```text
job-fraud-shield/
│
├── app.py                             # Main 5-tab interactive Streamlit dashboard
├── api.py                             # Production FastAPI REST microservice (/predict, /batch-predict)
├── cli.py                             # Command-Line Interface for automated & terminal inference
├── model.py                           # Standalone ML pipeline (TF-IDF + Balanced LR + Naive Bayes)
├── heuristics.py                      # Deterministic rule engine for high-severity scam patterns
├── test_pipeline.py                   # Automated test suite (8 benchmark cases, 100% pass)
├── generate_visualizations.py         # Visual chart generation pipeline
├── package_screenshots.py             # Screenshot packaging script
├── train_and_evaluate.py              # Initial training & model evaluation script
├── export_pipeline.py                 # Serializes models, vectorizer, and test samples
├── model_comparison_results.csv       # Benchmark metrics across models (Accuracy, F1, Recall, AUC)
├── test_results_summary.csv           # Test execution results matrix
├── requirements.txt                   # Dependency specifications
├── Dockerfile                         # Production container specification
├── docker-compose.yml                 # Multi-service container orchestration
├── .dockerignore                      # Docker ignore rules
├── LICENSE                            # MIT Open Source License
├── .gitignore                         # Git ignore rules for cache & large raw data
├── README.md                          # Comprehensive project documentation
│
├── models/                            # Serialized ML artifacts & precomputed insights
│   ├── tfidf_vectorizer.joblib        # Fitted TF-IDF Vectorizer (10,000 n-grams)
│   ├── lr_balanced.joblib             # Chosen model: Logistic Regression (Balanced weights)
│   ├── lr_default.joblib              # Standard Logistic Regression
│   ├── naive_bayes.joblib             # Multinomial Naive Bayes classifier
│   ├── sample_jobs.json               # 8 pre-loaded benchmark postings for UI testing
│   └── dataset_insights.json          # Precomputed dataset EDA & indicator metrics
│
├── screenshots/                       # High-resolution dashboard UI figures
│   ├── realtime_scanner_analysis.png  # Tab 1: Real-time scan with token attribution
│   ├── dataset_distribution_eda.png   # Tab 2: Imbalance donut, metadata flags, missing rates
│   ├── important_text_features.png    # Tab 3: Top predictive fraud vs legitimate n-grams
│   ├── model_performance_benchmarks.png # Tab 4: Evaluation bar chart & confusion matrix
│   └── common_warning_indicators.png  # Tab 5: Red flag prevalence & risk multipliers
│
└── archive/                           # Raw dataset storage (ignored in git)
    └── DataSet.csv                    # EMScAD 17,880-row dataset
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, feature requests, and bug reports are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
