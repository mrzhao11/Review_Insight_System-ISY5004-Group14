"""Prepare cleaned and sampled datasets from Amazon Reviews 2023 JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from src.utils.amazon_loader import (
    attach_metadata,
    build_meta_index,
    infer_category_from_path,
    load_jsonl,
    normalize_meta_record,
    normalize_review_record,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(
        description=(
            "Load Amazon Reviews 2023 JSONL files, apply cleaning/filtering, "
            "sample per category, and save processed train/validation/test data."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help=(
            "Directory containing downloaded Amazon JSONL files, such as "
            "All_Beauty.jsonl and meta_All_Beauty.jsonl."
        ),
    )
    parser.add_argument(
        "--review-file",
        type=Path,
        help="Optional path to a single review JSONL file.",
    )
    parser.add_argument(
        "--meta-file",
        type=Path,
        help="Optional path to a single metadata JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where cleaned sampled outputs will be written.",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=3000,
        help="Maximum number of cleaned reviews to keep for each category.",
    )
    parser.add_argument(
        "--min-review-words",
        type=int,
        default=5,
        help="Filter out reviews shorter than this many words after cleaning.",
    )
    parser.add_argument(
        "--min-review-chars",
        type=int,
        default=20,
        help="Filter out reviews shorter than this many characters after cleaning.",
    )
    parser.add_argument(
        "--only-verified",
        action="store_true",
        help="Keep only verified-purchase reviews.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional category whitelist, e.g. All_Beauty Electronics.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio for sampled records.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio for sampled records.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio for sampled records.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and split assignment.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional line limit for quick local tests.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before running the pipeline."""
    if not args.raw_dir and not args.review_file:
        raise ValueError("Provide either --raw-dir or --review-file.")

    if args.samples_per_category <= 0:
        raise ValueError("--samples-per-category must be greater than 0.")

    ratio_sum = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/validation/test ratios must sum to 1.0.")


def prepare_dataset(args: argparse.Namespace) -> None:
    """Run the end-to-end data preparation pipeline."""
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_dir:
        prepare_from_directory(args)
        return

    prepare_single_category(args)


def prepare_from_directory(args: argparse.Namespace) -> None:
    """Process all detected category files from a raw input directory."""
    review_files = discover_review_files(args.raw_dir, args.categories)
    if not review_files:
        raise ValueError(
            f"No review JSONL files found in {args.raw_dir}. "
            "Expected files such as All_Beauty.jsonl."
        )

    all_reviews: List[Dict[str, Any]] = []
    all_meta: Dict[str, Dict[str, Any]] = {}
    category_stats: Dict[str, Dict[str, Any]] = {}

    for review_file in review_files:
        category = infer_category_from_path(review_file)
        meta_file = find_meta_file(args.raw_dir, category)
        result = process_category(
            review_file=review_file,
            meta_file=meta_file,
            category=category,
            args=args,
        )
        all_reviews.extend(result["reviews"])
        all_meta.update(build_meta_index(result["meta"]))
        category_stats[category] = result["stats"]

    write_processed_bundle(
        reviews=all_reviews,
        meta_records=list(all_meta.values()),
        category_stats=category_stats,
        args=args,
        input_source=str(args.raw_dir),
    )


def prepare_single_category(args: argparse.Namespace) -> None:
    """Process one category from explicit review/meta file paths."""
    category = None
    if args.review_file:
        category = infer_category_from_path(args.review_file)
    elif args.meta_file:
        category = infer_category_from_path(args.meta_file)

    result = process_category(
        review_file=args.review_file,
        meta_file=args.meta_file,
        category=category or "unknown",
        args=args,
    )

    write_processed_bundle(
        reviews=result["reviews"],
        meta_records=result["meta"],
        category_stats={category or "unknown": result["stats"]},
        args=args,
        input_source=str(args.review_file or args.meta_file),
    )


def discover_review_files(
    raw_dir: Path,
    categories: list[str] | None = None,
) -> list[Path]:
    """Find review files in a raw directory, excluding metadata files."""
    category_filter = set(categories or [])
    files: list[Path] = []

    for path in sorted(raw_dir.glob("*.jsonl")):
        if path.name.startswith("meta_"):
            continue

        category = infer_category_from_path(path)
        if category_filter and category not in category_filter:
            continue

        files.append(path)

    return files


def find_meta_file(raw_dir: Path, category: str) -> Path | None:
    """Return the matching metadata file for a category if it exists."""
    candidate = raw_dir / f"meta_{category}.jsonl"
    if candidate.exists():
        return candidate

    for path in sorted(raw_dir.glob("meta_*.jsonl")):
        if infer_category_from_path(path) == category:
            return path

    return None


def process_category(
    *,
    review_file: Path | None,
    meta_file: Path | None,
    category: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Prepare sampled review and metadata records for a single category."""
    sampled_reviews, stats = sample_reviews_for_category(
        review_file=review_file,
        category=category,
        args=args,
    )

    split_reviews = assign_splits(sampled_reviews, category=category, args=args)
    selected_parent_asins = {
        review["parent_asin"]
        for review in split_reviews
        if review.get("parent_asin")
    }
    meta_records = load_selected_meta(
        meta_file=meta_file,
        category=category,
        parent_asins=selected_parent_asins,
        args=args,
    )

    stats["meta_file_found"] = bool(meta_file and meta_file.exists())
    stats["meta_records_selected"] = len(meta_records)
    stats["split_counts"] = count_by_split(split_reviews)

    return {
        "reviews": split_reviews,
        "meta": meta_records,
        "stats": stats,
    }


def sample_reviews_for_category(
    *,
    review_file: Path | None,
    category: str,
    args: argparse.Namespace,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Reservoir-sample cleaned review records for one category."""
    if not review_file:
        return [], empty_stats()

    rng = random.Random(f"sample:{args.seed}:{category}")
    reservoir: list[Dict[str, Any]] = []
    eligible_count = 0
    stats = empty_stats()
    stats["source_review_file"] = str(review_file)

    for raw_record in load_jsonl(review_file, limit=args.limit):
        stats["raw_records"] += 1
        review = normalize_review_record(
            raw_record,
            category=category,
            source_file=str(review_file),
        )
        reason = filter_review_record(review, args)
        if reason is not None:
            stats["filtered_records"] += 1
            stats["filter_reasons"][reason] = stats["filter_reasons"].get(reason, 0) + 1
            continue

        eligible_count += 1
        if len(reservoir) < args.samples_per_category:
            reservoir.append(review)
            continue

        replacement_index = rng.randint(0, eligible_count - 1)
        if replacement_index < args.samples_per_category:
            reservoir[replacement_index] = review

    stats["eligible_records"] = eligible_count
    stats["sampled_records"] = len(reservoir)
    stats["summary_target_records"] = sum(
        1 for review in reservoir if review.get("has_summary_target")
    )
    return reservoir, stats


def filter_review_record(
    review: Dict[str, Any],
    args: argparse.Namespace,
) -> str | None:
    """Return a filter reason if the review should be dropped."""
    if not review.get("product_id"):
        return "missing_product_id"

    if review.get("rating") is None:
        return "missing_rating"

    if not review.get("clean_review_text"):
        return "empty_review_text"

    if review.get("review_text_word_count", 0) < args.min_review_words:
        return "too_few_words"

    if review.get("review_text_char_count", 0) < args.min_review_chars:
        return "too_few_characters"

    if args.only_verified and not review.get("verified_purchase", False):
        return "not_verified_purchase"

    return None


def load_selected_meta(
    *,
    meta_file: Path | None,
    category: str,
    parent_asins: set[str],
    args: argparse.Namespace,
) -> list[Dict[str, Any]]:
    """Load only metadata rows that correspond to sampled reviews."""
    if not meta_file or not meta_file.exists() or not parent_asins:
        return []

    meta_records: list[Dict[str, Any]] = []
    for raw_record in load_jsonl(meta_file, limit=args.limit):
        parent_asin = raw_record.get("parent_asin")
        if parent_asin not in parent_asins:
            continue

        meta_records.append(
            normalize_meta_record(
                raw_record,
                category=category,
                source_file=str(meta_file),
            )
        )

    return meta_records


def assign_splits(
    reviews: Iterable[Dict[str, Any]],
    *,
    category: str,
    args: argparse.Namespace,
) -> list[Dict[str, Any]]:
    """Assign train/validation/test splits after sampling."""
    records = [dict(review) for review in reviews]
    rng = random.Random(f"split:{args.seed}:{category}")
    rng.shuffle(records)

    train_count, validation_count, test_count = compute_split_counts(
        total=len(records),
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    split_boundaries = {
        "train": train_count,
        "validation": train_count + validation_count,
        "test": train_count + validation_count + test_count,
    }

    split_records: list[Dict[str, Any]] = []
    for index, record in enumerate(records):
        updated_record = dict(record)
        if index < split_boundaries["train"]:
            updated_record["split"] = "train"
        elif index < split_boundaries["validation"]:
            updated_record["split"] = "validation"
        else:
            updated_record["split"] = "test"
        split_records.append(updated_record)

    return split_records


def compute_split_counts(
    *,
    total: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    """Compute robust split counts even for small sampled categories."""
    if total <= 0:
        return 0, 0, 0

    if total == 1:
        return 1, 0, 0

    if total == 2:
        return 1, 0, 1

    if total < 10:
        return total - 1, 0, 1

    validation_count = int(total * validation_ratio)
    test_count = int(total * test_ratio)

    if validation_ratio > 0 and validation_count == 0:
        validation_count = 1
    if test_ratio > 0 and test_count == 0:
        test_count = 1

    train_count = total - validation_count - test_count

    while train_count < 1 and validation_count > 0:
        validation_count -= 1
        train_count = total - validation_count - test_count

    while train_count < 1 and test_count > 0:
        test_count -= 1
        train_count = total - validation_count - test_count

    if train_count < 1:
        train_count = total
        validation_count = 0
        test_count = 0

    return train_count, validation_count, test_count


def count_by_split(reviews: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Count how many reviews were assigned to each split."""
    counts = {"train": 0, "validation": 0, "test": 0}
    for review in reviews:
        split = review.get("split")
        if split in counts:
            counts[split] += 1
    return counts


def write_processed_bundle(
    *,
    reviews: list[Dict[str, Any]],
    meta_records: list[Dict[str, Any]],
    category_stats: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    input_source: str,
) -> None:
    """Write processed reviews, splits, metadata, and a manifest."""
    deduped_meta = list(build_meta_index(meta_records).values())
    meta_index = build_meta_index(deduped_meta)
    enriched_reviews = list(attach_metadata(reviews, meta_index))

    sampled_reviews_path = args.output_dir / "sampled_reviews.jsonl"
    train_reviews_path = args.output_dir / "train_reviews.jsonl"
    validation_reviews_path = args.output_dir / "validation_reviews.jsonl"
    test_reviews_path = args.output_dir / "test_reviews.jsonl"
    summary_candidates_path = args.output_dir / "summary_candidates.jsonl"
    sampled_meta_path = args.output_dir / "sampled_meta.jsonl"
    enriched_reviews_path = args.output_dir / "sampled_reviews_with_meta.jsonl"
    manifest_path = args.output_dir / "dataset_manifest.json"

    write_jsonl(reviews, sampled_reviews_path)
    write_jsonl(
        (review for review in reviews if review.get("split") == "train"),
        train_reviews_path,
    )
    write_jsonl(
        (review for review in reviews if review.get("split") == "validation"),
        validation_reviews_path,
    )
    write_jsonl(
        (review for review in reviews if review.get("split") == "test"),
        test_reviews_path,
    )
    write_jsonl(
        (review for review in reviews if review.get("has_summary_target")),
        summary_candidates_path,
    )
    write_jsonl(deduped_meta, sampled_meta_path)
    write_jsonl(enriched_reviews, enriched_reviews_path)

    manifest = {
        "input_source": input_source,
        "config": {
            "samples_per_category": args.samples_per_category,
            "min_review_words": args.min_review_words,
            "min_review_chars": args.min_review_chars,
            "only_verified": args.only_verified,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "limit": args.limit,
            "categories": args.categories,
        },
        "totals": {
            "sampled_reviews": len(reviews),
            "train_reviews": sum(1 for review in reviews if review.get("split") == "train"),
            "validation_reviews": sum(
                1 for review in reviews if review.get("split") == "validation"
            ),
            "test_reviews": sum(1 for review in reviews if review.get("split") == "test"),
            "summary_candidates": sum(
                1 for review in reviews if review.get("has_summary_target")
            ),
            "sampled_meta_records": len(deduped_meta),
        },
        "categories": category_stats,
        "files": {
            "sampled_reviews": str(sampled_reviews_path),
            "train_reviews": str(train_reviews_path),
            "validation_reviews": str(validation_reviews_path),
            "test_reviews": str(test_reviews_path),
            "summary_candidates": str(summary_candidates_path),
            "sampled_meta": str(sampled_meta_path),
            "sampled_reviews_with_meta": str(enriched_reviews_path),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "Processed dataset ready: "
        f"{len(reviews)} sampled reviews across {len(category_stats)} category files."
    )
    print(f"Manifest written to {manifest_path}")


def empty_stats() -> Dict[str, Any]:
    """Return the default per-category stats dictionary."""
    return {
        "raw_records": 0,
        "eligible_records": 0,
        "filtered_records": 0,
        "sampled_records": 0,
        "summary_target_records": 0,
        "meta_file_found": False,
        "meta_records_selected": 0,
        "filter_reasons": {},
        "split_counts": {"train": 0, "validation": 0, "test": 0},
        "source_review_file": None,
    }


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main()
