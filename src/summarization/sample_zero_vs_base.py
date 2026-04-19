"""Generate a small qualitative zero-shot vs base comparison sample."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.summarization.train_t5 import build_prompt, rouge_l_f1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample a few test reviews and compare zero-shot Flan-T5-small with "
            "the fine-tuned Flan-T5-base student."
        )
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/processed/pseudo_summary_test.csv"),
        help="Pseudo-title test CSV.",
    )
    parser.add_argument(
        "--review-id-column",
        type=str,
        default="review_id",
        help="Review identifier column.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="clean_review_text",
        help="Input review text column.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="llm_complaint_title",
        help="Reference complaint title column.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="How many random rows to compare.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--zero-model-name",
        type=str,
        default="google/flan-t5-small",
        help="Zero-shot model name.",
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=Path("models/t5_pseudo_summary_base"),
        help="Fine-tuned base model directory.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("models/zero_vs_base_3sample_comparison.json"),
        help="Where to save the sampled comparison output.",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=256,
        help="Maximum input token length.",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=24,
        help="Maximum generated title length.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam size for decoding.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading zero-shot model files if not cached locally.",
    )
    return parser.parse_args()


def load_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Load valid rows from the pseudo-title test CSV."""
    rows: List[Dict[str, str]] = []
    with args.test_file.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            review_id = (row.get(args.review_id_column) or "").strip()
            text = (row.get(args.text_column) or "").strip()
            target = (row.get(args.target_column) or "").strip()
            if not text or not target:
                continue
            rows.append(
                {
                    "review_id": review_id,
                    "review_text": text,
                    "reference_title": target,
                }
            )
    return rows


def load_model_bundle(
    source: str | Path,
    *,
    local_files_only: bool,
) -> tuple[T5Tokenizer, T5ForConditionalGeneration, torch.device]:
    """Load tokenizer/model and move to the best local device."""
    tokenizer = T5Tokenizer.from_pretrained(source, local_files_only=local_files_only)
    model = T5ForConditionalGeneration.from_pretrained(source, local_files_only=local_files_only)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_title(
    text: str,
    *,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    device: torch.device,
    max_input_length: int,
    max_output_length: int,
    num_beams: int,
) -> str:
    """Generate a complaint title for one review."""
    prompt = build_prompt(text)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_length=max_output_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def run_sample_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    """Create a small random qualitative comparison set."""
    rows = load_rows(args)
    if len(rows) < args.sample_size:
        raise ValueError(
            f"Not enough rows in {args.test_file}. Requested {args.sample_size}, found {len(rows)}."
        )

    sampled = random.Random(args.seed).sample(rows, k=args.sample_size)

    zero_bundle = load_model_bundle(
        args.zero_model_name,
        local_files_only=not args.allow_download,
    )
    base_bundle = load_model_bundle(
        args.base_model_dir,
        local_files_only=True,
    )
    zero_tokenizer, zero_model, zero_device = zero_bundle
    base_tokenizer, base_model, base_device = base_bundle

    comparisons: List[Dict[str, Any]] = []
    for row in sampled:
        zero_title = generate_title(
            row["review_text"],
            tokenizer=zero_tokenizer,
            model=zero_model,
            device=zero_device,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            num_beams=args.num_beams,
        )
        base_title = generate_title(
            row["review_text"],
            tokenizer=base_tokenizer,
            model=base_model,
            device=base_device,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            num_beams=args.num_beams,
        )
        reference = row["reference_title"]
        zero_rouge_l = rouge_l_f1(zero_title, reference)
        base_rouge_l = rouge_l_f1(base_title, reference)
        if abs(base_rouge_l - zero_rouge_l) < 1e-9:
            winner = "tie"
        elif base_rouge_l > zero_rouge_l:
            winner = "base"
        else:
            winner = "zero"

        comparisons.append(
            {
                "review_id": row["review_id"],
                "review_text": row["review_text"],
                "reference_title": reference,
                "zero_shot_title": zero_title,
                "base_finetuned_title": base_title,
                "rougeL_f1_zero": zero_rouge_l,
                "rougeL_f1_base": base_rouge_l,
                "winner_by_rougeL": winner,
            }
        )

    base_wins = sum(1 for item in comparisons if item["winner_by_rougeL"] == "base")
    zero_wins = sum(1 for item in comparisons if item["winner_by_rougeL"] == "zero")
    ties = sum(1 for item in comparisons if item["winner_by_rougeL"] == "tie")

    result = {
        "config": {
            "test_file": str(args.test_file),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "zero_model_name": args.zero_model_name,
            "base_model_dir": str(args.base_model_dir),
            "scoring_note": "winner is decided by per-example ROUGE-L F1 only",
        },
        "summary": {
            "base_wins": base_wins,
            "zero_wins": zero_wins,
            "ties": ties,
        },
        "samples": comparisons,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = run_sample_comparison(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
