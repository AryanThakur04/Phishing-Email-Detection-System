"""
train.py
Trains a Random Forest classifier on the email dataset,
evaluates it, saves the model, and generates visualizations.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       roc_auc_score, roc_curve, accuracy_score,
                                       precision_score, recall_score, f1_score)
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────
BASE    = os.path.join(os.path.dirname(__file__))
DATA    = os.path.join(BASE, 'data',    'emails.csv')
MODELS  = os.path.join(BASE, 'models')
REPORTS = os.path.join(BASE, 'reports')
os.makedirs(MODELS,  exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

PALETTE = {
    'bg':      '#0a0c10',
    'surface': '#111318',
    'green':   '#00ff9d',
    'red':     '#ff4f6a',
    'blue':    '#4f9eff',
    'amber':   '#ffb547',
    'text':    '#e8eaf0',
    'muted':   '#6b7280',
}

plt.rcParams.update({
    'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['surface'],
    'axes.edgecolor':    '#1a1d25',
    'axes.labelcolor':   PALETTE['text'],
    'xtick.color':       PALETTE['muted'],
    'ytick.color':       PALETTE['muted'],
    'text.color':        PALETTE['text'],
    'grid.color':        '#1a1d25',
    'grid.linewidth':    0.8,
    'font.family':       'monospace',
    'figure.dpi':        130,
})


def load_data():
    df = pd.read_csv(DATA)
    X  = df[FEATURE_NAMES]
    y  = df['label']
    return X, y, df


def train_models(X_train, y_train):
    models = {
        'Random Forest': Pipeline([
            ('clf', RandomForestClassifier(
                n_estimators=200, max_depth=12,
                min_samples_split=4, random_state=42, n_jobs=-1
            ))
        ]),
        'Gradient Boosting': Pipeline([
            ('clf', GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1,
                max_depth=5, random_state=42
            ))
        ]),
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
    }
    trained = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe
        print(f"  ✓ {name} trained")
    return trained


def evaluate(models, X_test, y_test):
    results = {}
    for name, pipe in models.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            'accuracy':  round(accuracy_score(y_test, y_pred)  * 100, 2),
            'precision': round(precision_score(y_test, y_pred) * 100, 2),
            'recall':    round(recall_score(y_test, y_pred)    * 100, 2),
            'f1':        round(f1_score(y_test, y_pred)        * 100, 2),
            'auc':       round(roc_auc_score(y_test, y_prob)   * 100, 2),
            'cm':        confusion_matrix(y_test, y_pred),
            'y_pred':    y_pred,
            'y_prob':    y_prob,
        }
    return results


# ── Plot helpers ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['surface'])
    labels = [['True Neg\n(Legit)', 'False Pos\n(Legit→Phish)'],
              ['False Neg\n(Phish→Legit)', 'True Pos\n(Phish)']]
    colors = [[PALETTE['blue'], PALETTE['amber']],
              [PALETTE['red'],  PALETTE['green']]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                         color=colors[i][j], alpha=0.25))
            ax.text(j, i, f'{cm[i, j]}\n{labels[i][j]}',
                    ha='center', va='center', fontsize=9,
                    color=PALETTE['text'])
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nLegit', 'Predicted\nPhish'], fontsize=9)
    ax.set_yticklabels(['Actual\nLegit', 'Actual\nPhish'], fontsize=9)
    ax.set_title(title, fontsize=11, color=PALETTE['green'], pad=10)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_roc_curves(models_dict, X_test, y_test, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['surface'])
    cols = [PALETTE['green'], PALETTE['blue'], PALETTE['amber']]
    for (name, pipe), col in zip(models_dict.items(), cols):
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=col, lw=2, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1], '--', color=PALETTE['muted'], lw=1, label='Random')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models', color=PALETTE['green'], fontsize=11)
    ax.legend(fontsize=8, facecolor=PALETTE['surface'],
              edgecolor=PALETTE['muted'], labelcolor=PALETTE['text'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_feature_importance(rf_pipe, path):
    rf  = rf_pipe.named_steps['clf']
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]
    names = [FEATURE_NAMES[i] for i in idx]
    vals  = imp[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['surface'])
    bars = ax.barh(names[::-1], vals[::-1],
                   color=[PALETTE['green'] if v > 0.1 else
                          PALETTE['amber'] if v > 0.05 else
                          PALETTE['blue'] for v in vals[::-1]],
                   height=0.65)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=8, color=PALETTE['muted'])
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance — Random Forest', color=PALETTE['green'], fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results, path):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    names   = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.25
    cols    = [PALETTE['green'], PALETTE['blue'], PALETTE['amber']]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['surface'])

    for i, (name, col) in enumerate(zip(names, cols)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i*width, vals, width, label=name, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7,
                    color=PALETTE['text'])

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(80, 103)
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Comparison', color=PALETTE['green'], fontsize=11)
    ax.legend(fontsize=8, facecolor=PALETTE['surface'],
              edgecolor=PALETTE['muted'], labelcolor=PALETTE['text'])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_cross_val(models_dict, X, y, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['surface'])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cols = [PALETTE['green'], PALETTE['blue'], PALETTE['amber']]

    for i, (name, pipe) in enumerate(models_dict.items()):
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1', n_jobs=-1)
        ax.scatter([i]*5, scores*100, color=cols[i], s=50, zorder=3, alpha=0.8)
        ax.plot([i-0.2, i+0.2], [scores.mean()*100]*2,
                color=cols[i], lw=2.5)
        ax.text(i, scores.mean()*100 + 0.5, f'{scores.mean()*100:.1f}%',
                ha='center', va='bottom', fontsize=9, color=PALETTE['text'])

    ax.set_xticks(range(len(models_dict)))
    ax.set_xticklabels(list(models_dict.keys()), fontsize=9)
    ax.set_ylabel('F1 Score (%) per fold')
    ax.set_title('5-Fold Cross Validation', color=PALETTE['green'], fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*52)
    print("  PhishGuard — Model Training Pipeline")
    print("="*52)

    print("\n[1/5] Loading dataset...")
    X, y, df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[2/5] Training models...")
    models = train_models(X_train, y_train)

    print("\n[3/5] Evaluating models...")
    results = evaluate(models, X_test, y_test)
    best_name = max(results, key=lambda k: results[k]['f1'])

    print("\n  ┌──────────────────────┬────────┬────────┬────────┬────────┬────────┐")
    print("  │ Model                │ Acc    │ Prec   │ Recall │ F1     │ AUC    │")
    print("  ├──────────────────────┼────────┼────────┼────────┼────────┼────────┤")
    for name, r in results.items():
        tag = " ★" if name == best_name else "  "
        print(f"  │{tag}{name:<20}│{r['accuracy']:>6.2f}% │{r['precision']:>6.2f}% │"
              f"{r['recall']:>6.2f}% │{r['f1']:>6.2f}% │{r['auc']:>6.2f}% │")
    print("  └──────────────────────┴────────┴────────┴────────┴────────┴────────┘")
    print(f"\n  Best model: {best_name}")

    print("\n[4/5] Saving model...")
    best_pipe = models[best_name]
    with open(os.path.join(MODELS, 'phishguard_model.pkl'), 'wb') as f:
        pickle.dump({'model': best_pipe, 'name': best_name,
                     'features': FEATURE_NAMES}, f)
    print("      Saved → models/phishguard_model.pkl")

    print("\n[5/5] Generating report charts...")
    best_r = results[best_name]
    plot_confusion_matrix(best_r['cm'],
        f'Confusion Matrix — {best_name}',
        os.path.join(REPORTS, '1_confusion_matrix.png'))
    plot_roc_curves(models, X_test, y_test,
        os.path.join(REPORTS, '2_roc_curves.png'))
    plot_feature_importance(models['Random Forest'],
        os.path.join(REPORTS, '3_feature_importance.png'))
    plot_model_comparison(results,
        os.path.join(REPORTS, '4_model_comparison.png'))
    plot_cross_val(models, X, y,
        os.path.join(REPORTS, '5_cross_validation.png'))
    print("      Saved 5 charts → reports/")

    print("\n" + "="*52)
    print("  Training complete!")
    print("="*52 + "\n")
    return results, models


if __name__ == '__main__':
    main()
