"""Generate title-style summaries with a pretrained T5-family model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, List

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate concise review summaries with a pretrained T5 model."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/flan-t5-small",
        help="Pretrained T5-family checkpoint used for generation.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Optional local model directory. If provided, it overrides --model-name.",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single review text to summarize.",
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
        help="Maximum generated summary length.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam width for generation.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading model files if they are not already cached locally.",
    )
    return parser.parse_args()


def summarize_text(
    text: str,
    *,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    max_input_length: int,
    max_output_length: int,
    num_beams: int,
) -> str:
    """Generate a summary for a single review."""
    prefixed_text = (
        "Write a short complaint title for this customer review. "
        "Keep it concise and focused on the main problem.\n\n"
        f"Review: {text.strip()}"
    )
    inputs = tokenizer(
        prefixed_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = model.generate(
        **inputs,
        max_length=max_output_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def generate_summary(args: argparse.Namespace) -> Any:
    """Run single-text or batch summary generation."""
    model_source = args.model_dir if args.model_dir else args.model_name
    local_files_only = False if args.model_dir else not args.allow_download
    tokenizer = T5Tokenizer.from_pretrained(
        model_source,
        local_files_only=local_files_only,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_source,
        local_files_only=local_files_only,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.text:
        summary = summarize_text(
            args.text,
            tokenizer=tokenizer,
            model=model,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length,
            num_beams=args.num_beams,
        )
        return {"summary": summary}

    if not args.input_file:
        raise ValueError("Provide either --text or --input-file.")

    predictions: List[dict[str, str]] = []
    with args.input_file.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            text = (row.get(args.text_column) or "").strip()
            if not text:
                continue

            predictions.append(
                {
                    "text": text,
                    "summary": summarize_text(
                        text,
                        tokenizer=tokenizer,
                        model=model,
                        max_input_length=args.max_input_length,
                        max_output_length=args.max_output_length,
                        num_beams=args.num_beams,
                    ),
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
    result = generate_summary(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
