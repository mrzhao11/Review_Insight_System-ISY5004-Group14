"""Fine-tune a T5-family complaint-title generator on pseudo labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from src.summarization.train_t5 import build_prompt, evaluate_pairs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a T5 model using Ark-generated complaint-title labels."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/pseudo_summary_train.csv"),
        help="Path to pseudo-summary training CSV.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/pseudo_summary_validation.csv"),
        help="Path to pseudo-summary validation CSV.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/processed/pseudo_summary_test.csv"),
        help="Path to pseudo-summary test CSV.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="clean_review_text",
        help="Review text column used as source input.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="llm_complaint_title",
        help="Pseudo complaint-title column used as target text.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/flan-t5-small",
        help="Base T5-family checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/t5_pseudo_summary"),
        help="Where to save the fine-tuned model.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("models/t5_pseudo_summary_metrics.json"),
        help="Where to save training and evaluation metrics.",
    )
    parser.add_argument(
        "--samples-output",
        type=Path,
        default=Path("models/t5_pseudo_summary_samples.json"),
        help="Where to save decoded validation/test examples.",
    )
    parser.add_argument(
        "--input-max-length",
        type=int,
        default=256,
        help="Maximum source sequence length.",
    )
    parser.add_argument(
        "--target-max-length",
        type=int,
        default=24,
        help="Maximum target sequence length.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=5.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Trainer logging interval.",
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
        help="Allow downloading model files if not already cached locally.",
    )
    parser.add_argument(
        "--bertscore-model-type",
        type=str,
        default="distilbert-base-uncased",
        help=(
            "Backbone used to compute BERTScore F1. Use an empty string to skip "
            "BERTScore."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pairs(file_path: Path, *, text_column: str, target_column: str) -> List[Dict[str, str]]:
    """Load source/target pairs from a pseudo-summary CSV file."""
    pairs: List[Dict[str, str]] = []
    with file_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            review_text = (row.get(text_column) or "").strip()
            target_text = (row.get(target_column) or "").strip()
            if review_text and target_text:
                pairs.append({"review_text": review_text, "target_text": target_text})
    return pairs


class SummaryDataset(torch.utils.data.Dataset):
    """Seq2Seq dataset for complaint-title generation."""

    def __init__(
        self,
        pairs: List[Dict[str, str]],
        *,
        tokenizer: T5Tokenizer,
        input_max_length: int,
        target_max_length: int,
    ) -> None:
        prompts = [build_prompt(pair["review_text"]) for pair in pairs]
        targets = [pair["target_text"] for pair in pairs]
        model_inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=input_max_length,
        )
        labels = tokenizer(
            text_target=targets,
            truncation=True,
            max_length=target_max_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        self.encodings = model_inputs

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            key: torch.tensor(value[idx], dtype=torch.long)
            for key, value in self.encodings.items()
        }


def evaluate_model(
    pairs: List[Dict[str, str]],
    *,
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    input_max_length: int,
    target_max_length: int,
    eval_batch_size: int,
    sample_count: int,
    device: torch.device,
    bertscore_model_type: str | None,
) -> tuple[Dict[str, float], List[Dict[str, str]]]:
    """Generate titles and compute lightweight overlap metrics."""
    return evaluate_pairs(
        pairs,
        tokenizer=tokenizer,
        model=model,
        input_max_length=input_max_length,
        max_output_length=target_max_length,
        num_beams=4,
        batch_size=eval_batch_size,
        sample_count=sample_count,
        device=device,
        bertscore_model_type=bertscore_model_type,
    )


def fine_tune_pseudo_t5(args: argparse.Namespace) -> Dict[str, Any]:
    """Fine-tune and evaluate a pseudo-labeled T5 complaint-title generator."""
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.samples_output.parent.mkdir(parents=True, exist_ok=True)

    train_pairs = load_pairs(
        args.train_file,
        text_column=args.text_column,
        target_column=args.target_column,
    )
    validation_pairs = load_pairs(
        args.validation_file,
        text_column=args.text_column,
        target_column=args.target_column,
    )
    test_pairs = load_pairs(
        args.test_file,
        text_column=args.text_column,
        target_column=args.target_column,
    )
    if not train_pairs:
        raise ValueError("No training pairs found. Generate pseudo labels first.")

    tokenizer = T5Tokenizer.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_download,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_download,
    )

    train_dataset = SummaryDataset(
        train_pairs,
        tokenizer=tokenizer,
        input_max_length=args.input_max_length,
        target_max_length=args.target_max_length,
    )
    validation_dataset = SummaryDataset(
        validation_pairs,
        tokenizer=tokenizer,
        input_max_length=args.input_max_length,
        target_max_length=args.target_max_length,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        eval_strategy="epoch" if validation_pairs else "no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        predict_with_generate=True,
        generation_max_length=args.target_max_length,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset if validation_pairs else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    validation_trainer_metrics = (
        trainer.evaluate(validation_dataset) if validation_pairs else {}
    )

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    validation_metrics, validation_samples = evaluate_model(
        validation_pairs,
        tokenizer=tokenizer,
        model=model,
        input_max_length=args.input_max_length,
        target_max_length=args.target_max_length,
        eval_batch_size=args.eval_batch_size,
        sample_count=args.sample_count,
        device=device,
        bertscore_model_type=args.bertscore_model_type,
    )
    test_metrics, test_samples = evaluate_model(
        test_pairs,
        tokenizer=tokenizer,
        model=model,
        input_max_length=args.input_max_length,
        target_max_length=args.target_max_length,
        eval_batch_size=args.eval_batch_size,
        sample_count=args.sample_count,
        device=device,
        bertscore_model_type=args.bertscore_model_type,
    )

    metrics = {
        "config": {
            "train_file": str(args.train_file),
            "validation_file": str(args.validation_file),
            "test_file": str(args.test_file),
            "text_column": args.text_column,
            "target_column": args.target_column,
            "model_name": args.model_name,
            "output_dir": str(args.output_dir),
            "input_max_length": args.input_max_length,
            "target_max_length": args.target_max_length,
            "num_train_epochs": args.num_train_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "allow_download": args.allow_download,
            "bertscore_model_type": args.bertscore_model_type,
            "fine_tuned": True,
            "teacher_label_type": "ark_generated_pseudo_title",
        },
        "dataset_sizes": {
            "train": len(train_pairs),
            "validation": len(validation_pairs),
            "test": len(test_pairs),
        },
        "train": {k: float(v) for k, v in train_result.metrics.items()},
        "validation_trainer": {
            k: float(v)
            for k, v in validation_trainer_metrics.items()
            if isinstance(v, (int, float))
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.samples_output.write_text(
        json.dumps(
            {"validation": validation_samples, "test": test_samples},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    metrics = fine_tune_pseudo_t5(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
