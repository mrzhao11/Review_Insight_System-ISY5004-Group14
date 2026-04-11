"""Run review-value predictions with a trained sklearn pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict review-value labels from text with a trained model."
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path("models/review_value_classifier.pkl"),
        help="Path to the saved sklearn pipeline.",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single review text to classify.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Optional CSV file containing a clean_review_text column.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="clean_review_text",
        help="Text column name when using --input-file.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Optional path to save batch predictions as JSONL.",
    )
    return parser.parse_args()


def predict_helpfulness(args: argparse.Namespace) -> Any:
    """Run single-text or batch prediction."""
    model_bundle = joblib.load(args.model_file)
    if isinstance(model_bundle, dict):
        pipeline = model_bundle["pipeline"]
        decision_threshold = float(model_bundle.get("decision_threshold", 0.5))
    else:
        pipeline = model_bundle
        decision_threshold = 0.5

    if args.text:
        probability = float(pipeline.predict_proba([args.text])[0][1])
        label = int(probability >= decision_threshold)
        return {
            "review_value_label": label,
            "review_value_probability": probability,
            "decision_threshold": decision_threshold,
        }

    if not args.input_file:
        raise ValueError("Provide either --text or --input-file.")

    predictions = []
    with args.input_file.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            text = (row.get(args.text_column) or "").strip()
            if not text:
                continue

            probability = float(pipeline.predict_proba([text])[0][1])
            predictions.append(
                {
                    "text": text,
                    "review_value_label": int(probability >= decision_threshold),
                    "review_value_probability": probability,
                    "decision_threshold": decision_threshold,
                }
            )

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as fp:
            for record in predictions:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    return predictions


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = predict_helpfulness(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
