"""Build a review-value dataset for helpfulness classification."""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List

from src.preprocessing.prepare_dataset import (
    assign_splits,
    count_by_split,
    filter_review_record,
)
from src.utils.amazon_loader import (
    infer_category_from_path,
    load_jsonl,
    normalize_meta_record,
    normalize_review_record,
)

AMAZON_REVIEW_CATEGORIES = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
]

HUGGINGFACE_REVIEW_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/{category}.jsonl"
)
HUGGINGFACE_META_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/meta_{category}.jsonl"
)

CSV_FIELDS = [
    "review_id",
    "category",
    "product_id",
    "asin",
    "parent_asin",
    "product_title",
    "split",
    "rating",
    "helpful_votes",
    "review_value_label",
    "verified_purchase",
    "review_text_word_count",
    "review_text_char_count",
    "clean_review_title",
    "clean_review_text",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample multiple Amazon review categories and build a cleaned dataset "
            "for helpfulness classification."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Optional local directory with downloaded review JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where helpfulness-ready outputs will be written.",
    )
    parser.add_argument(
        "--num-categories",
        type=int,
        default=5,
        help="How many categories to sample when --categories is not provided.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional explicit category list.",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=1200,
        help="Maximum number of review samples to keep for each category.",
    )
    parser.add_argument(
        "--min-helpful-per-category",
        type=int,
        default=200,
        help="Target minimum count of positive helpful reviews per category, if available.",
    )
    parser.add_argument(
        "--helpful-vote-threshold",
        type=int,
        default=2,
        help="Helpful vote threshold used to create the binary helpfulness label.",
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
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for category selection and sampling.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional record limit per category for quick tests.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.samples_per_category <= 0:
        raise ValueError("--samples-per-category must be greater than 0.")

    ratio_sum = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/validation/test ratios must sum to 1.0.")

    if args.min_helpful_per_category < 0:
        raise ValueError("--min-helpful-per-category cannot be negative.")


def prepare_helpfulness_dataset(args: argparse.Namespace) -> None:
    """Build the review-value dataset and save it to the processed directory."""
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    categories = select_categories(args)
    all_reviews: list[Dict[str, Any]] = []
    manifest_categories: Dict[str, Dict[str, Any]] = {}

    for category in categories:
        sampled_reviews, stats = sample_category_reviews(category=category, args=args)
        sampled_reviews = attach_product_titles(
            reviews=sampled_reviews,
            category=category,
            args=args,
            stats=stats,
        )
        split_reviews = assign_splits(sampled_reviews, category=category, args=args)
        all_reviews.extend(split_reviews)
        stats["split_counts"] = count_by_split(split_reviews)
        manifest_categories[category] = stats
        print(
            f"{category}: kept {len(split_reviews)} reviews "
            f"({stats['high_value_labels']} high-value / "
            f"{stats['low_value_labels']} low-value, "
            f"{stats['product_titles_attached']} titles attached)"
        )

    write_outputs(all_reviews, categories, manifest_categories, args)


def select_categories(args: argparse.Namespace) -> list[str]:
    """Select categories explicitly or sample them reproducibly."""
    if args.categories:
        return list(args.categories)

    rng = random.Random(args.seed)
    if args.num_categories > len(AMAZON_REVIEW_CATEGORIES):
        raise ValueError("Requested more categories than available in the official list.")

    return sorted(rng.sample(AMAZON_REVIEW_CATEGORIES, args.num_categories))


def sample_category_reviews(
    *,
    category: str,
    args: argparse.Namespace,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Sample review-value records for one category."""
    positive_reservoir: list[Dict[str, Any]] = []
    negative_reservoir: list[Dict[str, Any]] = []
    positive_seen = 0
    negative_seen = 0
    stats = {
        "source": category_source_label(category, args),
        "meta_source": category_meta_source_label(category, args),
        "raw_records": 0,
        "eligible_records": 0,
        "filtered_records": 0,
        "high_value_labels": 0,
        "low_value_labels": 0,
        "product_titles_attached": 0,
        "missing_product_titles": 0,
        "filter_reasons": {},
        "split_counts": {"train": 0, "validation": 0, "test": 0},
    }

    target_positive = min(args.min_helpful_per_category, args.samples_per_category)
    target_total = args.samples_per_category
    positive_rng = random.Random(f"positive:{args.seed}:{category}")
    negative_rng = random.Random(f"negative:{args.seed}:{category}")

    for raw_record in iter_review_records(category, args):
        stats["raw_records"] += 1
        review = normalize_review_record(
            raw_record,
            category=category,
            source_file=stats["source"],
        )
        reason = filter_review_record(review, args)
        if reason is not None:
            stats["filtered_records"] += 1
            stats["filter_reasons"][reason] = stats["filter_reasons"].get(reason, 0) + 1
            continue

        stats["eligible_records"] += 1
        review["review_value_label"] = int(
            review.get("helpful_votes", 0) >= args.helpful_vote_threshold
        )

        if review["review_value_label"] == 1:
            positive_seen += 1
            reservoir_add(
                positive_reservoir,
                review,
                seen_count=positive_seen,
                capacity=target_total,
                rng=positive_rng,
            )
        else:
            negative_seen += 1
            reservoir_add(
                negative_reservoir,
                review,
                seen_count=negative_seen,
                capacity=target_total,
                rng=negative_rng,
            )

    rng = random.Random(f"final:{args.seed}:{category}")
    rng.shuffle(positive_reservoir)
    rng.shuffle(negative_reservoir)

    selected_positive = positive_reservoir[:target_positive]
    remaining_slots = max(target_total - len(selected_positive), 0)
    selected_negative = negative_reservoir[:remaining_slots]

    remaining_slots -= len(selected_negative)
    if remaining_slots > 0 and len(positive_reservoir) > len(selected_positive):
        extra_positives = positive_reservoir[len(selected_positive) :]
        rng.shuffle(extra_positives)
        selected_positive.extend(extra_positives[:remaining_slots])

    final_reviews = selected_positive + selected_negative
    rng.shuffle(final_reviews)

    stats["high_value_labels"] = sum(
        1 for review in final_reviews if review["review_value_label"] == 1
    )
    stats["low_value_labels"] = sum(
        1 for review in final_reviews if review["review_value_label"] == 0
    )
    return final_reviews, stats


def reservoir_add(
    reservoir: list[Dict[str, Any]],
    record: Dict[str, Any],
    *,
    seen_count: int,
    capacity: int,
    rng: random.Random,
) -> None:
    """Add a record to a reservoir sample."""
    if len(reservoir) < capacity:
        reservoir.append(record)
        return

    replacement_index = rng.randint(0, seen_count - 1)
    if replacement_index < capacity:
        reservoir[replacement_index] = record


def iter_review_records(category: str, args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    """Yield raw review records from a local file or the official remote URL."""
    local_file = find_local_review_file(category, args.raw_dir)
    if local_file is not None:
        yield from load_jsonl(local_file, limit=args.limit)
        return

    yield from load_remote_jsonl(build_review_url(category), limit=args.limit)


def attach_product_titles(
    *,
    reviews: list[Dict[str, Any]],
    category: str,
    args: argparse.Namespace,
    stats: Dict[str, Any],
) -> list[Dict[str, Any]]:
    """Attach product titles from metadata records when available."""
    parent_asins = {
        review.get("parent_asin")
        for review in reviews
        if review.get("parent_asin")
    }
    if not parent_asins:
        return reviews

    meta_index = load_meta_index(category=category, parent_asins=parent_asins, args=args)
    enriched_reviews: list[Dict[str, Any]] = []
    attached_count = 0

    for review in reviews:
        enriched_review = dict(review)
        metadata = meta_index.get(review.get("parent_asin"))
        product_title = ""
        if metadata:
            product_title = metadata.get("product_title") or ""

        if product_title:
            attached_count += 1

        enriched_review["product_title"] = product_title
        enriched_reviews.append(enriched_review)

    stats["product_titles_attached"] = attached_count
    stats["missing_product_titles"] = len(reviews) - attached_count
    return enriched_reviews


def find_local_review_file(category: str, raw_dir: Path | None) -> Path | None:
    """Find a local review file for the given category if it exists."""
    if raw_dir is None or not raw_dir.exists():
        return None

    candidate = raw_dir / f"{category}.jsonl"
    if candidate.exists():
        return candidate

    for path in sorted(raw_dir.glob("*.jsonl")):
        if path.name.startswith("meta_"):
            continue
        if infer_category_from_path(path) == category:
            return path

    return None


def find_local_meta_file(category: str, raw_dir: Path | None) -> Path | None:
    """Find a local metadata file for the given category if it exists."""
    if raw_dir is None or not raw_dir.exists():
        return None

    candidate = raw_dir / f"meta_{category}.jsonl"
    if candidate.exists():
        return candidate

    for path in sorted(raw_dir.glob("meta_*.jsonl")):
        if infer_category_from_path(path) == category:
            return path

    return None


def load_remote_jsonl(url: str, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    """Stream JSONL records from a remote URL."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "review-insight-system/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        for line_number, raw_line in enumerate(response, start=1):
            if limit is not None and line_number > limit:
                break

            stripped = raw_line.decode("utf-8").strip()
            if not stripped:
                continue

            yield json.loads(stripped)


def load_meta_index(
    *,
    category: str,
    parent_asins: set[str],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    """Load metadata records for sampled products and index them by parent ASIN."""
    meta_index: Dict[str, Dict[str, Any]] = {}

    local_meta_file = find_local_meta_file(category, args.raw_dir)
    if local_meta_file is not None:
        iterator = load_jsonl(local_meta_file)
        source_file = str(local_meta_file)
    else:
        iterator = load_remote_jsonl(build_meta_url(category))
        source_file = build_meta_url(category)

    for raw_record in iterator:
        parent_asin = raw_record.get("parent_asin")
        if parent_asin not in parent_asins:
            continue

        meta_index[parent_asin] = normalize_meta_record(
            raw_record,
            category=category,
            source_file=source_file,
        )

        if len(meta_index) == len(parent_asins):
            break

    return meta_index


def build_review_url(category: str) -> str:
    """Build the official Hugging Face review URL for a category."""
    return HUGGINGFACE_REVIEW_URL.format(category=category)


def build_meta_url(category: str) -> str:
    """Build the official Hugging Face metadata URL for a category."""
    return HUGGINGFACE_META_URL.format(category=category)


def category_source_label(category: str, args: argparse.Namespace) -> str:
    """Return a human-readable source label for manifest metadata."""
    local_file = find_local_review_file(category, args.raw_dir)
    if local_file is not None:
        return str(local_file)
    return build_review_url(category)


def category_meta_source_label(category: str, args: argparse.Namespace) -> str:
    """Return a human-readable metadata source label for manifest metadata."""
    local_file = find_local_meta_file(category, args.raw_dir)
    if local_file is not None:
        return str(local_file)
    return build_meta_url(category)


def write_outputs(
    reviews: list[Dict[str, Any]],
    categories: list[str],
    category_stats: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """Write model-ready review-value datasets and their manifest."""
    train_csv_path = args.output_dir / "review_value_train.csv"
    validation_csv_path = args.output_dir / "review_value_validation.csv"
    test_csv_path = args.output_dir / "review_value_test.csv"
    manifest_path = args.output_dir / "review_value_manifest.json"

    write_csv(
        [review for review in reviews if review.get("split") == "train"],
        train_csv_path,
    )
    write_csv(
        [review for review in reviews if review.get("split") == "validation"],
        validation_csv_path,
    )
    write_csv(
        [review for review in reviews if review.get("split") == "test"],
        test_csv_path,
    )

    manifest = {
        "selected_categories": categories,
        "config": {
            "num_categories": len(categories),
            "samples_per_category": args.samples_per_category,
            "min_helpful_per_category": args.min_helpful_per_category,
            "helpful_vote_threshold": args.helpful_vote_threshold,
            "min_review_words": args.min_review_words,
            "min_review_chars": args.min_review_chars,
            "only_verified": args.only_verified,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "limit": args.limit,
        },
        "totals": {
            "selected_reviews": len(reviews),
            "train_reviews": sum(1 for review in reviews if review.get("split") == "train"),
            "validation_reviews": sum(
                1 for review in reviews if review.get("split") == "validation"
            ),
            "test_reviews": sum(1 for review in reviews if review.get("split") == "test"),
            "high_value_labels": sum(
                1 for review in reviews if review.get("review_value_label") == 1
            ),
            "low_value_labels": sum(
                1 for review in reviews if review.get("review_value_label") == 0
            ),
        },
        "categories": category_stats,
        "files": {
            "train_csv": str(train_csv_path),
            "validation_csv": str(validation_csv_path),
            "test_csv": str(test_csv_path),
        },
        "columns": CSV_FIELDS,
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Review-value dataset written to {args.output_dir}")
    print(f"Manifest written to {manifest_path}")


def write_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Write selected fields as CSV for model-friendly consumption."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in CSV_FIELDS})


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    prepare_helpfulness_dataset(args)


if __name__ == "__main__":
    main()
