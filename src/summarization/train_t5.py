"""Evaluate a pretrained T5-family summarizer without fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a zero-shot T5-style summarization baseline on negative "
            "review-title pairs without fine-tuning."
        )
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/sentiment_train.csv"),
        help="Path to the calibrated training CSV file.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/sentiment_validation.csv"),
        help="Path to the calibrated validation CSV file.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/processed/sentiment_test.csv"),
        help="Path to the calibrated test CSV file.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="clean_review_text",
        help="Name of the cleaned review text column.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="clean_review_title",
        help="Name of the target review title column.",
    )
    parser.add_argument(
        "--sentiment-column",
        type=str,
        default="sentiment_label",
        help="Column used to filter negative review rows.",
    )
    parser.add_argument(
        "--negative-label",
        type=str,
        default="negative",
        help="Sentiment label that marks negative reviews.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/flan-t5-small",
        help="Pretrained T5-family checkpoint used for zero-shot generation.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("models/t5_summary_metrics.json"),
        help="Path to save evaluation metrics.",
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=Path("models/t5_summary_samples.json"),
        help="Path to save decoded validation/test examples.",
    )
    parser.add_argument(
        "--input-max-length",
        type=int,
        default=256,
        help="Maximum source sequence length.",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=24,
        help="Maximum generated summary length.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam search width for generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size used during zero-shot generation.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="How many example generations to save per split.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading model files if they are not already cached locally.",
    )
    return parser.parse_args()


def load_negative_pairs(
    file_path: Path,
    *,
    text_column: str,
    target_column: str,
    sentiment_column: str,
    negative_label: str,
) -> List[Dict[str, str]]:
    """Load negative review/title pairs from a CSV file."""
    pairs: List[Dict[str, str]] = []
    with file_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if (row.get(sentiment_column) or "").strip() != negative_label:
                continue

            review_text = (row.get(text_column) or "").strip()
            review_title = (row.get(target_column) or "").strip()
            if not review_text or not review_title:
                continue

            pairs.append(
                {
                    "review_text": review_text,
                    "target_text": review_title,
                }
            )
    return pairs


def build_prompt(review_text: str) -> str:
    """Format a short-title summarization prompt."""
    return (
        "Write a short complaint title for this customer review. "
        "Keep it concise and focused on the main problem.\n\n"
        f"Review: {review_text}"
    )


def normalize_text(text: str) -> str:
    """Normalize decoded text for lightweight overlap scoring."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def unigram_f1(prediction: str, reference: str) -> float:
    """Compute a lightweight unigram F1 score."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def generate_batch(
    prompts: List[str],
    *,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    input_max_length: int,
    max_output_length: int,
    num_beams: int,
    device: torch.device,
) -> List[str]:
    """Generate summaries for one batch of prompts."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=input_max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_output_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    return [
        tokenizer.decode(output, skip_special_tokens=True).strip()
        for output in outputs
    ]


def evaluate_pairs(
    pairs: List[Dict[str, str]],
    *,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    input_max_length: int,
    max_output_length: int,
    num_beams: int,
    batch_size: int,
    sample_count: int,
    device: torch.device,
) -> tuple[Dict[str, float], List[Dict[str, str]]]:
    """Run zero-shot generation and compute lightweight metrics."""
    predictions: List[str] = []
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        prompts = [build_prompt(pair["review_text"]) for pair in batch_pairs]
        predictions.extend(
            generate_batch(
                prompts,
                tokenizer=tokenizer,
                model=model,
                input_max_length=input_max_length,
                max_output_length=max_output_length,
                num_beams=num_beams,
                device=device,
            )
        )

    exact_matches = []
    f1_scores = []
    generated_lengths = []
    reference_lengths = []
    samples: List[Dict[str, str]] = []

    for pair, prediction in zip(pairs, predictions):
        reference = pair["target_text"]
        normalized_prediction = normalize_text(prediction)
        normalized_reference = normalize_text(reference)
        exact_matches.append(
            1.0 if normalized_prediction == normalized_reference else 0.0
        )
        f1_scores.append(unigram_f1(prediction, reference))
        generated_lengths.append(len(normalized_prediction.split()))
        reference_lengths.append(len(normalized_reference.split()))

        if len(samples) < sample_count:
            samples.append(
                {
                    "input_text": build_prompt(pair["review_text"]),
                    "target_text": reference,
                    "generated_summary": prediction,
                }
            )

    metrics = {
        "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "avg_unigram_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "avg_generated_words": (
            float(np.mean(generated_lengths)) if generated_lengths else 0.0
        ),
        "avg_reference_words": (
            float(np.mean(reference_lengths)) if reference_lengths else 0.0
        ),
    }
    return metrics, samples


def run_t5_baseline(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a zero-shot T5 baseline and save its outputs."""
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.samples_output.parent.mkdir(parents=True, exist_ok=True)

    train_pairs = load_negative_pairs(
        args.train_file,
        text_column=args.text_column,
        target_column=args.target_column,
        sentiment_column=args.sentiment_column,
        negative_label=args.negative_label,
    )
    validation_pairs = load_negative_pairs(
        args.validation_file,
        text_column=args.text_column,
        target_column=args.target_column,
        sentiment_column=args.sentiment_column,
        negative_label=args.negative_label,
    )
    test_pairs = load_negative_pairs(
        args.test_file,
        text_column=args.text_column,
        target_column=args.target_column,
        sentiment_column=args.sentiment_column,
        negative_label=args.negative_label,
    )

    tokenizer = T5Tokenizer.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_download,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_download,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    validation_metrics, validation_samples = evaluate_pairs(
        validation_pairs,
        tokenizer=tokenizer,
        model=model,
        input_max_length=args.input_max_length,
        max_output_length=args.max_output_length,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
        device=device,
    )
    test_metrics, test_samples = evaluate_pairs(
        test_pairs,
        tokenizer=tokenizer,
        model=model,
        input_max_length=args.input_max_length,
        max_output_length=args.max_output_length,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
        device=device,
    )

    metrics = {
        "config": {
            "train_file": str(args.train_file),
            "validation_file": str(args.validation_file),
            "test_file": str(args.test_file),
            "text_column": args.text_column,
            "target_column": args.target_column,
            "sentiment_column": args.sentiment_column,
            "negative_label": args.negative_label,
            "model_name": args.model_name,
            "input_max_length": args.input_max_length,
            "max_output_length": args.max_output_length,
            "num_beams": args.num_beams,
            "batch_size": args.batch_size,
            "allow_download": args.allow_download,
            "fine_tuned": False,
        },
        "dataset_sizes": {
            "train": len(train_pairs),
            "validation": len(validation_pairs),
            "test": len(test_pairs),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    samples = {
        "validation": validation_samples,
        "test": test_samples,
    }
    args.samples_output.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    metrics = run_t5_baseline(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
