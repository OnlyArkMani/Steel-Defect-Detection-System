"""
Quick script to check if evaluation results were generated
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Projects\CV_SDT")
RESULTS_DIR = PROJECT_ROOT / "results"

print("\n" + "=" * 60)
print("CHECKING EVALUATION RESULTS")
print("=" * 60 + "\n")

# Check what files exist
expected_files = [
    'confusion_matrix.png',
    'confusion_matrix_normalized.png',
    'classification_report.txt',
    'per_class_metrics.png',
    'roc_auc_scores.txt',
    'evaluation_summary.json'
]

print("Files in results directory:")
for file in expected_files:
    file_path = RESULTS_DIR / file
    if file_path.exists():
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} (missing)")

print("\n" + "=" * 60)

# If summary exists, show it
summary_path = RESULTS_DIR / 'evaluation_summary.json'
if summary_path.exists():
    print("EVALUATION SUMMARY")
    print("=" * 60 + "\n")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Print the raw summary to see structure
    print("Summary Contents:")
    print(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)

# Show classification report if exists
report_path = RESULTS_DIR / 'classification_report.txt'
if report_path.exists():
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60 + "\n")
    with open(report_path, 'r') as f:
        print(f.read())

print("\n✓ Check complete!")
print(f"View visualizations in: {RESULTS_DIR}\n")