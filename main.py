"""
main.py — PhishGuard full pipeline entry point.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generate_dataset import generate_dataset
from train            import main as train_main
from predict          import load_model, predict_email, print_result

def run_pipeline():
    print("\n" + "█"*52)
    print("  PhishGuard — Phishing Email Detection System")
    print("  Python + Machine Learning")
    print("█"*52)

    print("\n◈ STEP 1 — Generating email dataset...")
    os.makedirs('data', exist_ok=True)
    df = generate_dataset(n_phish=1000, n_legit=1000)
    df.to_csv('data/emails.csv', index=False)
    print(f"  Generated {len(df)} emails ({df['label'].sum()} phishing, {(df['label']==0).sum()} legit)")

    print("\n◈ STEP 2 — Training & evaluating models...")
    results, models = train_main()

    print("◈ STEP 3 — Live email predictions\n")
    model, model_name, feature_names = load_model()

    test_cases = [
        {'label':'⚠  Known phishing','sender':'security@paypa1-support.xyz',
         'subject':'URGENT: Your PayPal account SUSPENDED',
         'body':'Dear Valued Customer,\n\nYour account is SUSPENDED.\nVerify: http://192.168.1.45/paypal-login.php\nEnter password and credit card NOW.'},
        {'label':'✓  Legitimate email','sender':'orders@amazon.in',
         'subject':'Your order #402-1923847 has shipped',
         'body':'Hi,\n\nYour order has shipped.\nExpected delivery: Tomorrow.\nTrack at amazon.in/orders\n\nThank you!\nAmazon'},
        {'label':'⚠  Microsoft phishing','sender':'it@micros0ft-helpdesk.tk',
         'subject':'ACTION REQUIRED: Microsoft 365 Expires TODAY!!!',
         'body':'Your Microsoft 365 expires TODAY.\nRenew: http://ms-renew.ru/login\nEnter your username and password now.'},
        {'label':'✓  GitHub newsletter','sender':'newsletter@github.com',
         'subject':'Your GitHub digest for June 2024',
         'body':'Hi there,\n\nHere is your monthly GitHub digest.\nPRs merged: 7 | Issues closed: 12\nVisit github.com\n\nThe GitHub Team'},
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"  {'─'*50}")
        print(f"  Test #{i} — {case['label']}")
        print(f"  From    : {case['sender']}")
        print(f"  Subject : {case['subject']}")
        result = predict_email(case['sender'], case['subject'], case['body'], model, feature_names)
        print_result(result)

    print("◈ DONE — Reports saved to reports/ | Model saved to models/\n")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_pipeline()
