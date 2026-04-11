"""Fine-tune a BERT sentiment classifier with Hugging Face Trainer."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a BERT sentiment classifier on calibrated CSV splits."
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
        "--label-column",
        type=str,
        default="sentiment_target",
        help="Name of the integer sentiment target column.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained BERT checkpoint name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/bert_sentiment"),
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("models/bert_sentiment_metrics.json"),
        help="Path to save training and evaluation metrics.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=2.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=20,
        help="Trainer logging interval.",
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


def load_records(
    file_path: Path,
    *,
    text_column: str,
    label_column: str,
) -> List[Dict[str, Any]]:
    """Load text-label records from a CSV file."""
    records: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            text = (row.get(text_column) or "").strip()
            label = row.get(label_column)
            if not text or label in (None, ""):
                continue

            records.append(
                {
                    "text": text,
                    "label": int(label),
                }
            )
    return records


class SentimentDataset(torch.utils.data.Dataset):
    """Simple torch dataset for Trainer-compatible text classification."""

    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[idx], dtype=torch.long)
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_dataset(
    records: List[Dict[str, Any]],
    *,
    tokenizer: BertTokenizer,
    max_length: int,
) -> SentimentDataset:
    """Tokenize review text and build a torch dataset."""
    texts = [record["text"] for record in records]
    labels = [int(record["label"]) for record in records]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return SentimentDataset(encodings=encodings, labels=labels)


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Compute evaluation metrics for binary classification."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def train_sentiment_model(args: argparse.Namespace) -> Dict[str, Any]:
    """Fine-tune and evaluate a BERT sentiment classifier."""
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    train_records = load_records(
        args.train_file,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    validation_records = load_records(
        args.validation_file,
        text_column=args.text_column,
        label_column=args.label_column,
    )
    test_records = load_records(
        args.test_file,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    train_dataset = build_dataset(
        train_records,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    validation_dataset = build_dataset(
        validation_records,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    test_dataset = build_dataset(
        test_records,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    validation_metrics = trainer.evaluate(validation_dataset)
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = {
        "config": {
            "train_file": str(args.train_file),
            "validation_file": str(args.validation_file),
            "test_file": str(args.test_file),
            "text_column": args.text_column,
            "label_column": args.label_column,
            "model_name": args.model_name,
            "output_dir": str(args.output_dir),
            "max_length": args.max_length,
            "num_train_epochs": args.num_train_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
        },
        "dataset_sizes": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "train": {k: float(v) for k, v in train_result.metrics.items()},
        "validation": {
            k: float(v) for k, v in validation_metrics.items() if isinstance(v, (int, float))
        },
        "test": {
            k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))
        },
    }

    args.metrics_output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    metrics = train_sentiment_model(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
