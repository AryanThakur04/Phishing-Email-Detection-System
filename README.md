# PhishGuard — Phishing Email Detection System
**Python + Machine Learning**

## Project Structure
```
phishguard/
├── main.py                  ← Run this to execute full pipeline
├── requirements.txt
├── data/
│   └── emails.csv           ← 2000 labeled emails (auto-generated)
├── models/
│   └── phishguard_model.pkl ← Trained Random Forest model
├── reports/
│   ├── 1_confusion_matrix.png
│   ├── 2_roc_curves.png
│   ├── 3_feature_importance.png
│   ├── 4_model_comparison.png
│   └── 5_cross_validation.png
└── src/
    ├── generate_dataset.py  ← Synthetic email dataset generator
    ├── features.py          ← Feature extraction (14 signals)
    ├── train.py             ← Train 3 models + generate charts
    └── predict.py           ← Inference engine + CLI tester
```

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## Predict a Single Email
```python
from src.predict import load_model, predict_email, print_result

model, name, features = load_model()
result = predict_email(
    sender  = "security@paypa1-support.xyz",
    subject = "URGENT: Account Suspended",
    body    = "Dear Customer, verify now: http://192.168.1.1/login",
    model   = model,
    feature_names = features
)
print_result(result)
```

## Models
- Random Forest (200 trees) — best performer
- Gradient Boosting (150 estimators)
- Logistic Regression (baseline)

## 14 Extracted Features
url_count, has_ip_url, has_suspicious_tld, urgency_score,
credential_request, sender_subdomains, brand_spoof,
sender_has_numbers, subject_caps_ratio, subject_length,
exclamation_count, body_length, html_tags, greeting_generic
