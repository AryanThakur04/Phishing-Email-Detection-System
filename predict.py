"""
predict.py
Load the trained model and run predictions on new emails.
"""

import os
import sys
import pickle
import pandas as pd


sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features





BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE, 'models')

def load_model():
    model_path = os.path.join(MODELS, "phishguard_model.pkl")

    print("Looking for model at:", model_path)   # add this line for debugging

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run `python train.py` first.")

    import pickle
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    model_name = data['name']
    feature_names = data['features']

    return model, model_name, feature_names


def predict_email(sender: str, subject: str, body: str,
                  model=None, feature_names=None):
    """
    Predict whether an email is phishing.

    Returns
    -------
    dict with keys:
      verdict       — 'PHISHING' | 'SUSPICIOUS' | 'SAFE'
      phish_prob    — float 0–1
      confidence    — float 0–1
      features      — dict of extracted feature values
      top_flags     — list of (feature, value, risk) tuples
    """
    if model is None:
        model, _, feature_names = load_model()

    feats = extract_features(sender, subject, body)
    X     = pd.DataFrame([feats])[feature_names]
    prob  = model.predict_proba(X)[0][1]

    if prob >= 0.65:
        verdict    = 'PHISHING'
        confidence = prob
    elif prob >= 0.35:
        verdict    = 'SUSPICIOUS'
        confidence = 1 - abs(prob - 0.5) * 2
    else:
        verdict    = 'SAFE'
        confidence = 1 - prob

    # Top risk flags
    risk_map = {
        'has_ip_url':         ('IP address in URL',         feats['has_ip_url'],         'HIGH'),
        'brand_spoof':        ('Brand name spoofing',       feats['brand_spoof'],         'HIGH'),
        'urgency_score':      ('Urgency language score',    feats['urgency_score'],       'HIGH' if feats['urgency_score'] >= 3 else 'MED'),
        'credential_request': ('Credential request',        feats['credential_request'],  'HIGH' if feats['credential_request'] >= 2 else 'MED'),
        'has_suspicious_tld': ('Suspicious TLD in URL',     feats['has_suspicious_tld'],  'HIGH'),
        'url_count':          ('Number of URLs',            feats['url_count'],           'MED' if feats['url_count'] >= 2 else 'LOW'),
        'greeting_generic':   ('Generic greeting used',     feats['greeting_generic'],    'LOW'),
        'subject_caps_ratio': ('Subject CAPS ratio',        feats['subject_caps_ratio'],  'MED' if feats['subject_caps_ratio'] > 0.4 else 'LOW'),
        'sender_has_numbers': ('Numbers in sender domain',  feats['sender_has_numbers'],  'MED'),
    }

    top_flags = sorted(risk_map.values(), key=lambda x: {'HIGH': 0, 'MED': 1, 'LOW': 2}[x[2]])

    return {
        'verdict':     verdict,
        'phish_prob':  round(prob, 4),
        'confidence':  round(confidence, 4),
        'features':    feats,
        'top_flags':   top_flags,
    }


def print_result(result: dict):
    icons = {'PHISHING': '⚠  PHISHING', 'SUSPICIOUS': '◉  SUSPICIOUS', 'SAFE': '✓  SAFE'}
    colors = {'PHISHING': '\033[91m', 'SUSPICIOUS': '\033[93m', 'SAFE': '\033[92m'}
    RESET = '\033[0m'
    BOLD  = '\033[1m'
    DIM   = '\033[2m'

    v = result['verdict']
    print(f"\n  Verdict : {BOLD}{colors[v]}{icons[v]}{RESET}")
    print(f"  Phish probability : {result['phish_prob']*100:.1f}%")
    print(f"  Confidence        : {result['confidence']*100:.1f}%")
    print(f"\n  {DIM}── Feature Flags ────────────────────────────────{RESET}")
    for name, val, risk in result['top_flags']:
        risk_col = '\033[91m' if risk == 'HIGH' else '\033[93m' if risk == 'MED' else '\033[92m'
        print(f"  {risk_col}[{risk:4s}]{RESET}  {name:<32} {val}")
    print()


# ── CLI quick-test ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    model, model_name, feature_names = load_model()
    print(f"\n  PhishGuard Predictor — model: {model_name}\n")

    test_emails = [
        {
            'sender':  'security@paypa1-support.xyz',
            'subject': 'URGENT: Your account has been SUSPENDED',
            'body':    'Dear Valued Customer,\n\nYour PayPal account is SUSPENDED.\n'
                       'Verify immediately: http://192.168.1.45/login.php\n'
                       'Enter your password and credit card to restore access.\n\nAct NOW!',
        },
        {
            'sender':  'orders@amazon.in',
            'subject': 'Your order #402-1923847 has shipped',
            'body':    'Hi,\n\nYour order has shipped and will arrive tomorrow.\n'
                       'Track at amazon.in/orders\n\nThank you!\nAmazon',
        },
        {
            'sender':  'noreply@micros0ft-helpdesk.tk',
            'subject': 'Microsoft 365 License Expiring Today - Act Now',
            'body':    'Your Microsoft 365 expires today.\n'
                       'Renew: http://ms-renew.ru/login\nEnter your credentials now.',
        },
    ]

    for i, em in enumerate(test_emails, 1):
        print(f"  {'─'*50}")
        print(f"  Email #{i}")
        print(f"  From    : {em['sender']}")
        print(f"  Subject : {em['subject']}")
        result = predict_email(em['sender'], em['subject'], em['body'],
                               model, feature_names)
        print_result(result)
