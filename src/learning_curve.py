"""Does more data help? Learning curve on the TRAINING set only (ROC-AUC, 5-fold CV).

Run from the project root:  python -m src.learning_curve
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve

from src.config import FIGURES_DIR, RANDOM_SEED
from src.model_training import build_pipeline, load_split


def main() -> None:
    X_train, _, y_train, _ = load_split()   # test set is not touched
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, model in models.items():
        sizes, _, valid = learning_curve(
            build_pipeline(model), X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 8),
            cv=cv, scoring="roc_auc", n_jobs=1)
        mean = valid.mean(axis=1)
        ax.plot(sizes, mean, marker="o", label=name)
        print(f"{name}: " + ", ".join(f"{n}->{m:.3f}" for n, m in zip(sizes, mean)))

    ax.set_xlabel("Training rows")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_title("Does more data help?")
    ax.legend()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "18_learning_curve.png", bbox_inches="tight", dpi=120)
    print("Saved reports/figures/18_learning_curve.png")


if __name__ == "__main__":
    main()