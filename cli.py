"""
Job Fraud Shield - Command-Line Interface (CLI)
==============================================
Provides instant terminal-based fraud evaluation for job listings.

Usage:
  python cli.py "Paste complete job text here..."
  python cli.py --file path/to/job_posting.txt
  python cli.py --sample "Suspicious: Fake Check & Wire Transfer Clerk (Advance Fee)"
  python cli.py --threshold 0.40 --explain "..."
"""

import os
import sys
import json
import argparse
from model import JobFraudModelPipeline
from heuristics import scan_risk_indicators

def main():
    parser = argparse.ArgumentParser(
        description="Job Fraud Shield CLI: AI Scam & Fraudulent Job Posting Detector"
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="",
        help="Raw text of the job posting to analyze."
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default="",
        help="Path to a text file containing the job posting."
    )
    parser.add_argument(
        "--sample", "-s",
        type=str,
        default="",
        help="Name of a pre-loaded benchmark sample from sample_jobs.json."
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.50,
        help="Decision threshold for fraud classification (default: 0.50)."
    )
    parser.add_argument(
        "--explain", "-e",
        action="store_true",
        help="Show top token attributions driving the prediction."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results for piping to downstream tools."
    )

    args = parser.parse_args()

    # Load input
    job_text = ""
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    if args.sample:
        samples_path = os.path.join(models_dir, "sample_jobs.json")
        if os.path.exists(samples_path):
            with open(samples_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
            matching = [k for k in samples.keys() if args.sample.lower() in k.lower()]
            if matching:
                job_text = samples[matching[0]]
                if not args.json:
                    print(f"Loaded Sample: [{matching[0]}]\n")
            else:
                print(f"Error: Sample '{args.sample}' not found in sample_jobs.json. Available samples:")
                for k in samples.keys():
                    print(f"  - {k}")
                sys.exit(1)
        else:
            print("Error: sample_jobs.json not found.")
            sys.exit(1)
    elif args.file:
        if os.path.exists(args.file):
            with open(args.file, "r", encoding="utf-8") as f:
                job_text = f.read()
        else:
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
    elif args.text:
        job_text = args.text
    else:
        parser.print_help()
        sys.exit(0)

    # Initialize model
    pipeline = JobFraudModelPipeline(models_dir=models_dir)
    pipeline.load()

    # Inference
    pred = pipeline.predict(job_text, threshold=args.threshold)
    flags = scan_risk_indicators(job_text)
    has_critical = any(f['severity'] in ['HIGH', 'CRITICAL'] for f in flags)
    is_fraud = pred['is_fraud'] or has_critical

    if is_fraud:
        risk_level = "HIGH RISK (SUSPECTED SCAM)"
    elif pred['fraud_probability'] >= 0.25 or len(flags) > 0:
        risk_level = "MODERATE RISK (PROCEED WITH CAUTION)"
    else:
        risk_level = "LOW RISK (LIKELY LEGITIMATE)"

    explanation = pipeline.explain(job_text) if args.explain else None

    # JSON Output
    if args.json:
        out = {
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "fraud_probability": pred["fraud_probability"],
            "legitimate_probability": pred["legitimate_probability"],
            "heuristic_flags": flags,
            "flag_count": len(flags)
        }
        if explanation and not explanation["fraud_leaning"].empty:
            out["top_fraud_tokens"] = explanation["fraud_leaning"][["term", "contribution", "weight"]].to_dict(orient="records")
        print(json.dumps(out, indent=2))
        return

    # Clean Cross-Platform Terminal Output (ASCII compatible)
    print("=" * 70)
    print("JOB FRAUD SHIELD: PREDICTION REPORT")
    print("=" * 70)
    print(f"Risk Assessment:      {risk_level}")
    print(f"Fraud Probability:    {pred['fraud_probability'] * 100:.1f}%")
    print(f"Legit Probability:    {pred['legitimate_probability'] * 100:.1f}%")
    print(f"Decision Threshold:   {args.threshold * 100:.1f}%")
    print(f"Heuristic Red Flags:  {len(flags)} detected")
    print("-" * 70)

    if flags:
        print("\n[!] DETECTED HEURISTIC RED FLAGS:")
        for idx, fl in enumerate(flags, 1):
            print(f"  {idx}. [{fl['severity']}] {fl['category']}")
            print(f"     Matched: {', '.join(fl['matched'])}")
            print(f"     Tip:     {fl['tip']}")

    if explanation:
        print("\n[*] TOKEN-LEVEL FEATURE ATTRIBUTION:")
        f_df = explanation['fraud_leaning']
        if not f_df.empty:
            print("  Top Tokens Driving Fraud Prediction (+):")
            for _, r in f_df.head(6).iterrows():
                print(f"    + {r['term']:22s} | Contribution: +{r['contribution']:.3f} (Weight: {r['weight']:.2f})")
        l_df = explanation['legit_leaning']
        if not l_df.empty:
            print("  Top Tokens Driving Legitimate Prediction (-):")
            for _, r in l_df.head(6).iterrows():
                print(f"    - {r['term']:22s} | Contribution: {r['contribution']:.3f} (Weight: {r['weight']:.2f})")

    print("=" * 70)

if __name__ == "__main__":
    main()
