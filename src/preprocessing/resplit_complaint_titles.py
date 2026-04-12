"""Re-split Ark-generated complaint-title pseudo labels without new API calls."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-split generated complaint-title pseudo labels into larger "
            "validation/test sets without calling Ark again."
        )
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/pseudo_summary_train.csv"),
        help="Existing pseudo-summary train CSV.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/pseudo_summary_validation.csv"),
        help="Existing pseudo-summary validation CSV.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/processed/pseudo_summary_test.csv"),
        help="Existing pseudo-summary test CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where re-split pseudo-summary CSV files are written.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("data/processed/pseudo_summary_manifest.json"),
        help="Pseudo-summary manifest to update with re-split metadata.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train ratio for the re-split pseudo-title dataset.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.10,
        help="Validation ratio for the re-split pseudo-title dataset.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Test ratio for the re-split pseudo-title dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic re-splitting.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate split ratios."""
    ratio_sum = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/validation/test ratios must sum to 1.0.")
    if min(args.train_ratio, args.validation_ratio, args.test_ratio) < 0:
        raise ValueError("Split ratios cannot be negative.")


def load_rows(file_paths: List[Path]) -> tuple[List[Dict[str, Any]], List[str]]:
    """Load pseudo-title rows and preserve the input column order."""
    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []

    for file_path in file_paths:
        with file_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            if not fieldnames and reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            for row in reader:
                updated = dict(row)
                updated["source_split"] = row.get("split", "")
                rows.append(updated)

    if "source_split" not in fieldnames:
        fieldnames.append("source_split")
    return rows, fieldnames


def deduplicate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate rows by review_id while preserving first occurrence."""
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for row in rows:
        review_id = str(row.get("review_id", "")).strip()
        key = review_id or f"row:{len(deduped)}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def compute_split_counts(
    total: int,
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """Compute deterministic split counts with the test split prioritized."""
    if total <= 0:
        return 0, 0, 0
    if total < 5:
        test_count = 1 if test_ratio > 0 and total > 1 else 0
        validation_count = 1 if validation_ratio > 0 and total > 2 else 0
        return total - validation_count - test_count, validation_count, test_count

    test_count = int(total * test_ratio)
    validation_count = int(total * validation_ratio)
    if test_ratio > 0 and test_count == 0:
        test_count = 1
    if validation_ratio > 0 and validation_count == 0:
        validation_count = 1

    train_count = total - validation_count - test_count
    if train_count <= 0:
        raise ValueError("Split ratios leave no rows for training.")
    return train_count, validation_count, test_count


def assign_splits(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    """Shuffle and assign pseudo-title rows to new splits."""
    shuffled = [dict(row) for row in rows]
    random.Random(args.seed).shuffle(shuffled)
    train_count, validation_count, _ = compute_split_counts(
        len(shuffled),
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    split_rows = {"train": [], "validation": [], "test": []}
    for index, row in enumerate(shuffled):
        updated = dict(row)
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        updated["split"] = split
        split_rows[split].append(updated)
    return split_rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path, fieldnames: List[str]) -> None:
    """Write rows with a stable field order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_manifest(
    manifest_path: Path,
    *,
    files: Dict[str, str],
    split_rows: Dict[str, List[Dict[str, Any]]],
    rows_before_dedup: int,
    rows_after_dedup: int,
    args: argparse.Namespace,
) -> None:
    """Record re-split metadata in the pseudo-summary manifest."""
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    manifest["files"] = files
    manifest["resplit"] = {
        "strategy": "deterministic_shuffle",
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "rows_before_dedup": rows_before_dedup,
        "rows_after_dedup": rows_after_dedup,
        "split_counts": {
            split: len(rows) for split, rows in split_rows.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resplit_complaint_titles(args: argparse.Namespace) -> Dict[str, Any]:
    """Re-split pseudo-title rows and write train/validation/test CSV files."""
    validate_args(args)
    input_files = [args.train_file, args.validation_file, args.test_file]
    rows, fieldnames = load_rows(input_files)
    deduped_rows = deduplicate_rows(rows)
    split_rows = assign_splits(deduped_rows, args)

    files = {
        split: str(args.output_dir / f"pseudo_summary_{split}.csv")
        for split in ("train", "validation", "test")
    }
    for split, rows_for_split in split_rows.items():
        write_csv(rows_for_split, Path(files[split]), fieldnames)

    update_manifest(
        args.manifest_file,
        files=files,
        split_rows=split_rows,
        rows_before_dedup=len(rows),
        rows_after_dedup=len(deduped_rows),
        args=args,
    )
    result = {
        "rows_before_dedup": len(rows),
        "rows_after_dedup": len(deduped_rows),
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "files": files,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    resplit_complaint_titles(args)


if __name__ == "__main__":
    main()
