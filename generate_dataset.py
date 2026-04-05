"""
generate_dataset.py
Generates a realistic synthetic labeled email dataset for training.
"""

import pandas as pd
import numpy as np
import random
import re

random.seed(42)
np.random.seed(42)

# ── Phishing templates ──────────────────────────────────────────────────────
PHISHING_SUBJECTS = [
    "URGENT: Your account has been SUSPENDED",
    "Action Required: Verify your account immediately",
    "Your PayPal account is limited - Act Now",
    "Security Alert: Unusual login detected",
    "Final Notice: Update your billing information",
    "Your package could not be delivered - Click here",
    "Congratulations! You've won a $1000 gift card",
    "IMPORTANT: Your Netflix subscription will expire",
    "Confirm your identity or lose account access",
    "Your bank account has been compromised",
    "Microsoft account verification required",
    "Apple ID suspended - verify now",
    "IRS Tax Refund Notification",
    "You have a pending wire transfer",
    "Immediate action required on your account",
]

PHISHING_BODIES = [
    "Dear Valued Customer,\n\nWe have detected unusual activity on your account. "
    "Your account has been SUSPENDED.\n\nClick here to verify: http://192.168.1.45/login.php\n\n"
    "Enter your username, password, and credit card number to restore access.\n\nAct within 24 hours!",

    "Hello,\n\nYour PayPal account has been limited due to suspicious activity.\n"
    "To restore your account, please verify your information immediately:\n"
    "http://paypa1-secure.xyz/verify\n\nProvide your SSN and credit card to continue.",

    "Dear User,\n\nCongratulations! You have been selected for a $500 Amazon gift card.\n"
    "Click here to claim: http://amaz0n-gifts.tk/claim?id=WIN500\n\n"
    "This offer expires in 2 hours. Act NOW!",

    "URGENT NOTICE:\n\nYour Microsoft 365 subscription expires TODAY.\n"
    "Renew immediately at: http://ms-renew.ru/login\n\n"
    "Failure to act will result in permanent account deletion.\n\nEnter your credentials now.",

    "Dear Customer,\n\nWe noticed a login from an unknown device.\n"
    "Verify your identity here: http://103.24.56.78/secure/verify\n\n"
    "Confirm your password and banking PIN to secure your account.",
]

PHISHING_SENDERS = [
    "security@paypa1-support.com",
    "noreply@amaz0n-verify.xyz",
    "support@micros0ft-helpdesk.tk",
    "alerts@apple-id-verify.ru",
    "service@netfl1x-billing.ml",
    "admin@secure-bank-alert.ga",
    "irs-refund@taxnotification.xyz",
    "it-support@company-helpdesk.ru",
    "billing@account-update-now.tk",
    "verify@paypal-secure-login.xyz",
]

# ── Legitimate templates ────────────────────────────────────────────────────
LEGIT_SUBJECTS = [
    "Your GitHub monthly digest",
    "Invoice #INV-2024-0891 from Zoho",
    "Your Amazon order has shipped",
    "Meeting reminder: Team standup at 10am",
    "Weekly newsletter from Medium",
    "Your flight booking confirmation",
    "Password changed successfully",
    "New comment on your pull request",
    "Your subscription receipt",
    "Project update from Asana",
    "Welcome to the team!",
    "Your monthly bank statement is ready",
    "Scheduled maintenance notification",
    "Thanks for your purchase",
    "Your account summary for June",
]

LEGIT_BODIES = [
    "Hi there,\n\nHere is your GitHub digest for this month.\n"
    "Repositories starred: 12 | Pull requests merged: 5 | Issues closed: 8\n\n"
    "Visit github.com to see your full activity.\n\nThe GitHub Team",

    "Hello,\n\nYour Amazon order #402-1923847 has shipped.\n"
    "Expected delivery: Tomorrow by 9 PM.\n\n"
    "Track at amazon.in/orders\n\nThank you for shopping with us.",

    "Hi Team,\n\nJust a reminder about our daily standup meeting tomorrow at 10 AM IST.\n"
    "Agenda: sprint progress, blockers, and planning.\n\n"
    "Meeting link: meet.google.com/abc-defg-hij\n\nSee you there!",

    "Dear Customer,\n\nYour password was successfully changed on June 15, 2024.\n"
    "If you did not make this change, please contact support at support@company.com.\n\n"
    "No action is required if this was you.\n\nSecurity Team",

    "Hi,\n\nYour invoice #INV-2024-0891 for ₹4,999 is attached.\n"
    "Due date: July 1, 2024.\n\nPay securely at billing.zoho.com/invoice/0891\n\n"
    "Thank you for your business!\nZoho Finance",
]

LEGIT_SENDERS = [
    "newsletter@github.com",
    "orders@amazon.in",
    "noreply@google.com",
    "billing@zoho.com",
    "digest@medium.com",
    "noreply@netflix.com",
    "support@slack.com",
    "no-reply@linkedin.com",
    "notifications@asana.com",
    "team@notion.so",
]


def extract_features(sender, subject, body):
    """Extract ML features from a raw email."""
    text = (subject + " " + body).lower()
    full = (sender + " " + subject + " " + body).lower()

    features = {}

    # URL features
    urls = re.findall(r'https?://\S+', body, re.IGNORECASE)
    features['url_count'] = len(urls)
    features['has_ip_url'] = int(bool(re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', body)))
    features['has_suspicious_tld'] = int(bool(re.search(r'\.(xyz|tk|ru|ml|ga|pw|cf|gq)\b', body, re.IGNORECASE)))

    # Urgency signals
    urgency_words = ['urgent', 'immediately', 'suspended', 'act now', 'verify',
                     'expire', 'final warning', 'within 24', 'limited', 'restricted',
                     'compromised', 'click here', 'act fast', 'last chance']
    features['urgency_score'] = sum(1 for w in urgency_words if w in text)

    # Credential fishing
    cred_words = ['password', 'credit card', 'social security', 'ssn', 'pin',
                  'confirm your', 'enter your', 'billing info', 'bank account']
    features['credential_request'] = sum(1 for w in cred_words if w in text)

    # Sender features
    sender_domain = sender.split('@')[-1] if '@' in sender else ''
    features['sender_subdomains'] = sender_domain.count('.')
    features['brand_spoof'] = int(bool(re.search(r'paypa1|amaz0n|micros0ft|app1e|netfl1x|g00gle', full)))
    features['sender_has_numbers'] = int(bool(re.search(r'\d', sender_domain)))

    # Subject features
    features['subject_caps_ratio'] = round(
        sum(1 for c in subject if c.isupper()) / max(len(subject), 1), 2)
    features['subject_length'] = len(subject)
    features['exclamation_count'] = subject.count('!') + body.count('!')

    # Body features
    features['body_length'] = len(body)
    features['html_tags'] = len(re.findall(r'<[^>]+>', body))
    features['greeting_generic'] = int(bool(re.search(r'dear (valued|customer|user|member)', text)))

    return features


def generate_dataset(n_phish=1000, n_legit=1000):
    rows = []

    # Generate phishing emails
    for _ in range(n_phish):
        sender = random.choice(PHISHING_SENDERS)
        subject = random.choice(PHISHING_SUBJECTS)
        body = random.choice(PHISHING_BODIES)

        # Add slight variation
        if random.random() > 0.5:
            subject = subject.upper()
        if random.random() > 0.6:
            body += f"\n\nClick: http://{random.randint(10,250)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}/login"

        feats = extract_features(sender, subject, body)
        feats['label'] = 1  # phishing
        feats['sender'] = sender
        feats['subject'] = subject
        rows.append(feats)

    # Generate legit emails
    for _ in range(n_legit):
        sender = random.choice(LEGIT_SENDERS)
        subject = random.choice(LEGIT_SUBJECTS)
        body = random.choice(LEGIT_BODIES)

        feats = extract_features(sender, subject, body)
        feats['label'] = 0  # legitimate
        feats['sender'] = sender
        feats['subject'] = subject
        rows.append(feats)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset(1000, 1000)
    df.to_csv("data/emails.csv", index=False)
    print(f"Dataset saved: {len(df)} emails ({df['label'].sum()} phishing, {(df['label']==0).sum()} legit)")
    print(df.head())
