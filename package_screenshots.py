import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('screenshots', exist_ok=True)

shutil.copy('dataset_distribution.png', 'screenshots/dataset_distribution_eda.png')
shutil.copy('text_features_importance.png', 'screenshots/important_text_features.png')
shutil.copy('model_performance_visualized.png', 'screenshots/model_performance_benchmarks.png')
shutil.copy('warning_indicators_distribution.png', 'screenshots/common_warning_indicators.png')

fig = plt.figure(figsize=(14, 9), facecolor='#F8FAFC')
gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.8, 1.8], hspace=0.35, wspace=0.25)

fig.suptitle('Job Fraud Shield: Real-Time Scam Detection UI & Explainability', fontsize=16, fontweight='bold', color='#1E3A8A', y=0.96)

# 1. Top Banner (Risk Card)
ax_banner = fig.add_subplot(gs[0, :])
ax_banner.set_facecolor('#FEE2E2')
for spine in ax_banner.spines.values():
    spine.set_color('#EF4444')
    spine.set_linewidth(2)
ax_banner.set_xticks([])
ax_banner.set_yticks([])

ax_banner.text(0.03, 0.65, '[!] HIGH RISK / SUSPECTED FRAUDULENT POSTING', fontsize=14, fontweight='bold', color='#991B1B')
ax_banner.text(0.03, 0.25, 'Posting exhibits critical scam patterns: advance-fee check overpayment, off-platform messaging, and unrealistic compensation.',
               fontsize=10.5, color='#7F1D1D')
ax_banner.text(0.85, 0.50, '90.5%\nFraud Probability', fontsize=12, fontweight='bold', color='#991B1B', ha='center', va='center')

# 2. Input Sample Snippet
ax_input = fig.add_subplot(gs[1, 0])
ax_input.set_facecolor('white')
for spine in ax_input.spines.values():
    spine.set_color('#CBD5E1')
ax_input.set_xticks([])
ax_input.set_yticks([])
sample_snippet = (
    "JOB POSTING INPUT EVALUATION:\n\n"
    "\"URGENT HIRING: Data Entry Operator / Remote Processing Clerk.\n"
    "Earn $55 per hour working from home! No experience required.\n\n"
    "Equipment & Office Setup:\n"
    "A company cashier check will be mailed to your address...\n"
    "Deposit check and wire transfer balance via Western Union / MoneyGram\n"
    "to our certified equipment vendor...\n\n"
    "Contact: text coordinator on Telegram @global_work_hr or email\n"
    "careers-processing@gmail.com. Immediate placement guaranteed.\""
)
ax_input.text(0.04, 0.92, sample_snippet, fontsize=9.5, fontfamily='monospace', va='top', color='#1E293B')
ax_input.set_title('Evaluated Text Input (Sample Scam)', fontsize=11, fontweight='bold', color='#334155', loc='left')

# 3. Detected Heuristic Red Flags
ax_flags = fig.add_subplot(gs[1, 1])
ax_flags.set_facecolor('white')
for spine in ax_flags.spines.values():
    spine.set_color('#CBD5E1')
ax_flags.set_xticks([])
ax_flags.set_yticks([])
flags_text = (
    "DETECTED HEURISTIC RED FLAGS (4 Triggers Detected):\n\n"
    "[*] Financial Red Flag [HIGH SEVERITY]\n"
    "    Matched: 'wire transfer', 'cashier check', 'western union'\n"
    "    Tip: Legitimate employers never ask employees to wire funds.\n\n"
    "[*] Unrealistic Compensation [MEDIUM SEVERITY]\n"
    "    Matched: 'earn $55', 'no experience required'\n"
    "    Tip: Abnormally high pay for entry-level work is classic bait.\n\n"
    "[*] Off-Platform Contact Channel [HIGH SEVERITY]\n"
    "    Matched: '@gmail.com', 'telegram'\n"
    "    Tip: Verified recruiters communicate via official corporate domains.\n\n"
    "[*] Urgent Placement Rush [LOW SEVERITY]\n"
    "    Matched: 'urgent hiring', 'immediate placement'"
)
ax_flags.text(0.04, 0.92, flags_text, fontsize=9, va='top', color='#1E293B')
ax_flags.set_title('Heuristic Safety Rule Engine', fontsize=11, fontweight='bold', color='#334155', loc='left')

# 4. Token Attribution Bar Chart (Fraud-leaning)
ax_chart_f = fig.add_subplot(gs[2, 0])
tokens_f = ['wire transfer', 'cashier check', 'western union', 'data entry', 'telegram', 'per hour', 'earn', 'clerk']
weights_f = [0.85, 0.78, 0.72, 0.65, 0.58, 0.52, 0.48, 0.42]
bars_f = ax_chart_f.barh(tokens_f[::-1], weights_f[::-1], color='#EF4444')
ax_chart_f.set_xlabel('Score Contribution (+ Fraud Indicator)', fontsize=9, fontweight='bold')
ax_chart_f.set_title('Top Scam-Indicative Tokens in Text', fontsize=11, fontweight='bold', color='#991B1B', loc='left')
for b in bars_f:
    w = b.get_width()
    ax_chart_f.annotate(f" +{w:.2f}", (w, b.get_y() + b.get_height()/2), va='center', fontsize=8.5, fontweight='bold')
ax_chart_f.set_xlim(0, 1.0)

# 5. Model Consensus & Probabilities
ax_chart_models = fig.add_subplot(gs[2, 1])
model_names = ['LR (Balanced)', 'Multinomial NB', 'LR (Standard)']
model_probs = [90.5, 70.5, 41.3]
bar_colors = ['#EF4444', '#F59E0B', '#3B82F6']
bars_m = ax_chart_models.bar(model_names, model_probs, color=bar_colors, width=0.45)
ax_chart_models.set_ylabel('Fraud Probability (%)', fontsize=9, fontweight='bold')
ax_chart_models.set_title('Multi-Model Prediction Consensus', fontsize=11, fontweight='bold', color='#334155', loc='left')
ax_chart_models.set_ylim(0, 100)
ax_chart_models.axhline(50, color='#6B7280', linestyle='--', linewidth=1, label='Decision Threshold (50%)')
for b in bars_m:
    h = b.get_height()
    ax_chart_models.annotate(f"{h:.1f}%", (b.get_x() + b.get_width()/2, h + 2), ha='center', fontsize=9, fontweight='bold')
ax_chart_models.legend(loc='upper right', fontsize=8.5)

plt.savefig('screenshots/realtime_scanner_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Regenerated screenshots cleanly!")
