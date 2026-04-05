"""
features.py
Feature extraction engine for raw email input.
"""

import re


URGENCY_WORDS = [
    'urgent', 'immediately', 'suspended', 'act now', 'verify', 'expire',
    'final warning', 'within 24', 'limited', 'restricted', 'compromised',
    'click here', 'act fast', 'last chance', 'account closed', 'respond now'
]

CREDENTIAL_WORDS = [
    'password', 'credit card', 'social security', 'ssn', 'pin',
    'confirm your', 'enter your', 'billing info', 'bank account',
    'update payment', 'submit details'
]

SUSPICIOUS_TLDS = r'\.(xyz|tk|ru|ml|ga|pw|cf|gq|top|icu|win)\b'
BRAND_SPOOF_RE  = r'paypa1|amaz0n|micros0ft|app1e|netfl1x|g00gle|yah00|faceb00k'


def extract_features(sender: str, subject: str, body: str) -> dict:
    """
    Extract a feature vector from a raw email.

    Parameters
    ----------
    sender  : str  — From address
    subject : str  — Email subject
    body    : str  — Plain-text body

    Returns
    -------
    dict of feature_name → numeric value
    """
    text = (subject + " " + body).lower()
    full = (sender + " " + subject + " " + body).lower()

    # ── URL features ──────────────────────────────────────────────────
    urls = re.findall(r'https?://\S+', body, re.IGNORECASE)
    url_count = len(urls)
    has_ip_url = int(bool(
        re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', body)
    ))
    has_suspicious_tld = int(bool(
        re.search(SUSPICIOUS_TLDS, body, re.IGNORECASE)
    ))

    # ── Urgency / pressure language ───────────────────────────────────
    urgency_score = sum(1 for w in URGENCY_WORDS if w in text)

    # ── Credential fishing ────────────────────────────────────────────
    credential_request = sum(1 for w in CREDENTIAL_WORDS if w in text)

    # ── Sender analysis ───────────────────────────────────────────────
    sender_domain = sender.split('@')[-1] if '@' in sender else ''
    sender_subdomains  = sender_domain.count('.')
    brand_spoof        = int(bool(re.search(BRAND_SPOOF_RE, full)))
    sender_has_numbers = int(bool(re.search(r'\d', sender_domain)))

    # ── Subject analysis ──────────────────────────────────────────────
    subject_caps_ratio = round(
        sum(1 for c in subject if c.isupper()) / max(len(subject), 1), 3
    )
    subject_length    = len(subject)
    exclamation_count = subject.count('!') + body.count('!')

    # ── Body analysis ─────────────────────────────────────────────────
    body_length     = len(body)
    html_tags       = len(re.findall(r'<[^>]+>', body))
    greeting_generic = int(bool(
        re.search(r'dear (valued|customer|user|member)', text)
    ))

    return {
        'url_count':          url_count,
        'has_ip_url':         has_ip_url,
        'has_suspicious_tld': has_suspicious_tld,
        'urgency_score':      urgency_score,
        'credential_request': credential_request,
        'sender_subdomains':  sender_subdomains,
        'brand_spoof':        brand_spoof,
        'sender_has_numbers': sender_has_numbers,
        'subject_caps_ratio': subject_caps_ratio,
        'subject_length':     subject_length,
        'exclamation_count':  exclamation_count,
        'body_length':        body_length,
        'html_tags':          html_tags,
        'greeting_generic':   greeting_generic,
    }


FEATURE_NAMES = list(extract_features('a@b.com', 'test', 'test').keys())
