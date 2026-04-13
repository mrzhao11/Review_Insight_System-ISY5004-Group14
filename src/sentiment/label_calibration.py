"""Calibrate sentiment labels with rating signals and VADER lexicon scores."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a higher-precision sentiment dataset by combining rating labels "
            "with VADER lexicon scores."
        )
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/review_value_train.csv"),
        help="Path to the review-value training CSV file.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/review_value_validation.csv"),
        help="Path to the review-value validation CSV file.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/processed/review_value_test.csv"),
        help="Path to the review-value test CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where calibrated sentiment files will be saved.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="clean_review_text",
        help="Name of the cleaned review text column.",
    )
    parser.add_argument(
        "--rating-column",
        type=str,
        default="rating",
        help="Name of the rating column.",
    )
    parser.add_argument(
        "--value-label-column",
        type=str,
        default="review_value_label",
        help="Column used to identify high-value reviews.",
    )
    parser.add_argument(
        "--high-value-only",
        action="store_true",
        help="Use only high-value reviews for sentiment calibration.",
    )
    parser.add_argument(
        "--positive-score-threshold",
        type=float,
        default=0.0,
        help="Minimum VADER compound score for a positive label.",
    )
    parser.add_argument(
        "--negative-score-threshold",
        type=float,
        default=0.05,
        help="Maximum VADER compound score for a negative label.",
    )
    parser.add_argument(
        "--positive-rating-min",
        type=float,
        default=4.0,
        help="Minimum star rating that can be labeled as positive.",
    )
    parser.add_argument(
        "--negative-rating-max",
        type=float,
        default=2.0,
        help="Maximum star rating that can be labeled as negative.",
    )
    parser.add_argument(
        "--train-positive-negative-ratio",
        type=float,
        default=3.0,
        help=(
            "Downsample train positives to this ratio times the number of train "
            "negatives. Set to 0 to keep all positives."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when downsampling the train split.",
    )
    return parser.parse_args()


def lexicon_score(text: str, analyzer: SentimentIntensityAnalyzer) -> float:
    """Return the VADER compound sentiment score."""
    return float(analyzer.polarity_scores(text)["compound"])


def calibrate_label(
    rating: float,
    score: float,
    *,
    positive_rating_min: float,
    negative_rating_max: float,
    positive_score_threshold: float,
    negative_score_threshold: float,
) -> str:
    """Combine rating and lexicon score into a calibrated sentiment label."""
    if rating >= positive_rating_min and score > positive_score_threshold:
        return "positive"
    if rating <= negative_rating_max and score < negative_score_threshold:
        return "negative"
    return "discard"


def load_rows(file_path: Path) -> List[Dict[str, Any]]:
    """Load rows from a CSV file."""
    with file_path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def calibrate_rows(
    rows: List[Dict[str, Any]],
    *,
    analyzer: SentimentIntensityAnalyzer,
    text_column: str,
    rating_column: str,
    value_label_column: str,
    high_value_only: bool,
    positive_rating_min: float,
    negative_rating_max: float,
    positive_score_threshold: float,
    negative_score_threshold: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply lexicon scoring and label calibration to a list of review rows."""
    calibrated_rows: List[Dict[str, Any]] = []
    stats = {
        "input_rows": len(rows),
        "rows_seen_for_calibration": 0,
        "discarded_rows": 0,
        "positive_rows": 0,
        "negative_rows": 0,
    }

    for row in rows:
        if high_value_only and int(row.get(value_label_column, 0)) != 1:
            continue

        stats["rows_seen_for_calibration"] += 1
        text = (row.get(text_column) or "").strip()
        rating_raw = row.get(rating_column)

        if not text or rating_raw in (None, ""):
            stats["discarded_rows"] += 1
            continue

        rating = float(rating_raw)
        score = lexicon_score(text, analyzer)
        sentiment_label = calibrate_label(
            rating,
            score,
            positive_rating_min=positive_rating_min,
            negative_rating_max=negative_rating_max,
            positive_score_threshold=positive_score_threshold,
            negative_score_threshold=negative_score_threshold,
        )

        if sentiment_label == "discard":
            stats["discarded_rows"] += 1
            continue

        enriched_row = dict(row)
        enriched_row["lex_score"] = f"{score:.6f}"
        enriched_row["sentiment_label"] = sentiment_label
        enriched_row["sentiment_target"] = "1" if sentiment_label == "positive" else "0"
        calibrated_rows.append(enriched_row)

        if sentiment_label == "positive":
            stats["positive_rows"] += 1
        else:
            stats["negative_rows"] += 1

    return calibrated_rows, stats


def rebalance_train_rows(
    rows: List[Dict[str, Any]],
    *,
    max_positive_negative_ratio: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Downsample train positives so the dataset is less skewed."""
    stats = {
        "rows_before_rebalancing": len(rows),
        "rows_after_rebalancing": len(rows),
        "positive_rows_removed": 0,
        "positive_rows_before_rebalancing": sum(
            1 for row in rows if row["sentiment_label"] == "positive"
        ),
        "negative_rows_before_rebalancing": sum(
            1 for row in rows if row["sentiment_label"] == "negative"
        ),
    }

    if max_positive_negative_ratio <= 0:
        return rows, stats

    positives = [row for row in rows if row["sentiment_label"] == "positive"]
    negatives = [row for row in rows if row["sentiment_label"] == "negative"]

    if not positives or not negatives:
        return rows, stats

    max_positive_rows = int(len(negatives) * max_positive_negative_ratio)
    if len(positives) <= max_positive_rows:
        return rows, stats

    rng = random.Random(seed)
    kept_positives = rng.sample(positives, k=max_positive_rows)
    rebalanced_rows = kept_positives + negatives
    rng.shuffle(rebalanced_rows)

    stats["rows_after_rebalancing"] = len(rebalanced_rows)
    stats["positive_rows_removed"] = len(positives) - len(kept_positives)
    return rebalanced_rows, stats


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write calibrated rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calibrate_sentiment_labels(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a calibrated sentiment dataset from processed review-value files."""
    analyzer = SentimentIntensityAnalyzer()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": args.train_file,
        "validation": args.validation_file,
        "test": args.test_file,
    }

    manifest: Dict[str, Any] = {
        "config": {
            "train_file": str(args.train_file),
            "validation_file": str(args.validation_file),
            "test_file": str(args.test_file),
            "text_column": args.text_column,
            "rating_column": args.rating_column,
            "value_label_column": args.value_label_column,
            "high_value_only": args.high_value_only,
            "positive_rating_min": args.positive_rating_min,
            "negative_rating_max": args.negative_rating_max,
            "positive_score_threshold": args.positive_score_threshold,
            "negative_score_threshold": args.negative_score_threshold,
            "train_positive_negative_ratio": args.train_positive_negative_ratio,
            "seed": args.seed,
            "label_rules": {
                "positive": (
                    f"rating >= {args.positive_rating_min} and "
                    f"lex_score > {args.positive_score_threshold}"
                ),
                "negative": (
                    f"rating <= {args.negative_rating_max} and "
                    f"lex_score < {args.negative_score_threshold}"
                ),
                "discard": "otherwise",
            },
        },
        "splits": {},
        "files": {},
    }

    total_positive = 0
    total_negative = 0

    for split_name, input_file in split_map.items():
        rows = load_rows(input_file)
        calibrated_rows, stats = calibrate_rows(
            rows,
            analyzer=analyzer,
            text_column=args.text_column,
            rating_column=args.rating_column,
            value_label_column=args.value_label_column,
            high_value_only=args.high_value_only,
            positive_rating_min=args.positive_rating_min,
            negative_rating_max=args.negative_rating_max,
            positive_score_threshold=args.positive_score_threshold,
            negative_score_threshold=args.negative_score_threshold,
        )

        rebalance_stats: Optional[Dict[str, int]] = None
        if split_name == "train":
            calibrated_rows, rebalance_stats = rebalance_train_rows(
                calibrated_rows,
                max_positive_negative_ratio=args.train_positive_negative_ratio,
                seed=args.seed,
            )
            if rebalance_stats:
                stats.update(rebalance_stats)
                stats["positive_rows"] = sum(
                    1 for row in calibrated_rows if row["sentiment_label"] == "positive"
                )
                stats["negative_rows"] = sum(
                    1 for row in calibrated_rows if row["sentiment_label"] == "negative"
                )
        else:
            stats["rows_before_rebalancing"] = len(calibrated_rows)
            stats["rows_after_rebalancing"] = len(calibrated_rows)
            stats["positive_rows_removed"] = 0

        output_file = args.output_dir / f"sentiment_{split_name}.csv"
        write_csv(calibrated_rows, output_file)

        manifest["splits"][split_name] = stats
        manifest["files"][split_name] = str(output_file)
        total_positive += stats["positive_rows"]
        total_negative += stats["negative_rows"]

        print(
            f"{split_name}: kept {len(calibrated_rows)} rows "
            f"({stats['positive_rows']} positive / {stats['negative_rows']} negative)"
        )

    manifest["totals"] = {
        "positive_rows": total_positive,
        "negative_rows": total_negative,
        "kept_rows": total_positive + total_negative,
    }

    manifest_path = args.output_dir / "sentiment_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest written to {manifest_path}")
    return manifest


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    calibrate_sentiment_labels(args)


if __name__ == "__main__":
    main()
