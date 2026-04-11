"""Train the review-value classifier with TF-IDF plus Logistic Regression."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the review-value classifier on processed CSV splits."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/review_value_train.csv"),
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/review_value_validation.csv"),
        help="Path to the validation CSV file.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/processed/review_value_test.csv"),
        help="Path to the test CSV file.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="clean_review_text",
        help="Name of the text column to vectorize.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="review_value_label",
        help="Name of the binary target column.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/review_value_classifier.pkl"),
        help="Where to save the fitted sklearn pipeline.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("models/review_value_metrics.json"),
        help="Where to save evaluation metrics.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=1,
        help="Minimum document frequency for TF-IDF features.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum Logistic Regression iterations.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="liblinear",
        help="Logistic Regression solver.",
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=0.5,
        help="Inverse regularization strength for Logistic Regression.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        help="Class weight strategy. Use 'none' to disable balancing.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.6,
        help="Positive-class probability threshold used for classification.",
    )
    return parser.parse_args()


def load_dataset(
    file_path: Path,
    *,
    text_column: str,
    label_column: str,
) -> tuple[list[str], list[int]]:
    """Load texts and labels from a CSV file."""
    texts: List[str] = []
    labels: List[int] = []

    with file_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            text = (row.get(text_column) or "").strip()
            label = row.get(label_column)
            if not text or label is None or label == "":
                continue

            texts.append(text)
            labels.append(int(label))

    return texts, labels


def build_pipeline(
    max_features: int,
    max_iter: int,
    *,
    min_df: int,
    solver: str,
    c_value: float,
    class_weight: str | None,
) -> Pipeline:
    """Create the sklearn pipeline used for review-value classification."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=min_df,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=max_iter,
                    solver=solver,
                    C=c_value,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def evaluate_split(
    pipeline: Pipeline,
    texts: list[str],
    labels: list[int],
    *,
    decision_threshold: float,
) -> Dict[str, Any]:
    """Evaluate a fitted pipeline on one split."""
    probabilities = pipeline.predict_proba(texts)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    return {
        "size": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "decision_threshold": decision_threshold,
        "classification_report": report,
    }


def train_helpfulness_model(args: argparse.Namespace) -> Dict[str, Any]:
    """Train and evaluate the review-value classifier."""
    X_train, y_train = load_dataset(
        args.train_file,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    X_validation, y_validation = load_dataset(
        args.validation_file,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    X_test, y_test = load_dataset(
        args.test_file,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    class_weight = None if args.class_weight == "none" else args.class_weight
    pipeline = build_pipeline(
        args.max_features,
        args.max_iter,
        min_df=args.min_df,
        solver=args.solver,
        c_value=args.c_value,
        class_weight=class_weight,
    )
    pipeline.fit(X_train, y_train)

    metrics = {
        "config": {
            "train_file": str(args.train_file),
            "validation_file": str(args.validation_file),
            "test_file": str(args.test_file),
            "text_column": args.text_column,
            "label_column": args.label_column,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "ngram_range": [1, 2],
            "max_iter": args.max_iter,
            "solver": args.solver,
            "c_value": args.c_value,
            "class_weight": class_weight,
            "decision_threshold": args.decision_threshold,
        },
        "train": {"size": len(y_train)},
        "validation": evaluate_split(
            pipeline,
            X_validation,
            y_validation,
            decision_threshold=args.decision_threshold,
        ),
        "test": evaluate_split(
            pipeline,
            X_test,
            y_test,
            decision_threshold=args.decision_threshold,
        ),
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "pipeline": pipeline,
            "decision_threshold": args.decision_threshold,
            "text_column": args.text_column,
            "label_column": args.label_column,
        },
        args.model_output,
    )
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metrics


def print_summary(metrics: Dict[str, Any]) -> None:
    """Print a concise training summary."""
    validation_accuracy = metrics["validation"]["accuracy"]
    test_accuracy = metrics["test"]["accuracy"]

    print("Validation accuracy:", f"{validation_accuracy:.4f}")
    print("Test accuracy:", f"{test_accuracy:.4f}")
    print()
    print("Validation report:")
    print(
        json.dumps(
            metrics["validation"]["classification_report"],
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print("Test report:")
    print(
        json.dumps(
            metrics["test"]["classification_report"],
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    metrics = train_helpfulness_model(args)
    print_summary(metrics)


if __name__ == "__main__":
    main()
