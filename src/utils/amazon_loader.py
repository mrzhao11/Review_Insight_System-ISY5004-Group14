"""Utilities for loading Amazon Reviews 2023 JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

from src.preprocessing.clean_text import clean_text


def _safe_int(value: Any) -> int | None:
    """Convert a value to int when possible."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_category_from_path(file_path: str | Path) -> str:
    """Infer the dataset category from an Amazon Reviews filename."""
    path = Path(file_path)
    name = path.name

    if name.startswith("meta_"):
        name = name[len("meta_") :]

    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]

    for suffix in (".sample.review", ".sample.meta", ".sample", ".review", ".meta"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    return name


def load_jsonl(file_path: str | Path, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    """Yield records from a JSONL file line by line."""
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if limit is not None and line_number > limit:
                break

            stripped = line.strip()
            if not stripped:
                continue

            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSONL record in {path} at line {line_number}."
                ) from exc


def normalize_review_record(
    record: Dict[str, Any],
    *,
    category: str | None = None,
    source_file: str | None = None,
) -> Dict[str, Any]:
    """Normalize an Amazon review record into a project-friendly schema."""
    review_text = record.get("text", "")
    review_title = record.get("title", "")
    timestamp = record.get("timestamp")
    sort_timestamp = record.get("sort_timestamp", timestamp)
    helpful_votes = record.get("helpful_votes", record.get("helpful_vote", 0))
    clean_review_text = clean_text(review_text)
    clean_review_title = clean_text(review_title)

    normalized: Dict[str, Any] = {
        "review_id": build_review_id(record),
        "category": category,
        "dataset_version": "amazon_reviews_2023",
        "source_file": source_file,
        "product_id": record.get("parent_asin") or record.get("asin"),
        "asin": record.get("asin"),
        "parent_asin": record.get("parent_asin"),
        "user_id": record.get("user_id"),
        "rating": _safe_float(record.get("rating")),
        "helpful_votes": _safe_int(helpful_votes) or 0,
        "verified_purchase": bool(record.get("verified_purchase", False)),
        "timestamp": _safe_int(timestamp),
        "sort_timestamp": _safe_int(sort_timestamp),
        "review_title": review_title,
        "review_text": review_text,
        "clean_review_title": clean_review_title,
        "clean_review_text": clean_review_text,
        "review_text_char_count": len(clean_review_text),
        "review_text_word_count": len(clean_review_text.split()),
        "review_title_char_count": len(clean_review_title),
        "review_title_word_count": len(clean_review_title.split()),
        "has_summary_target": bool(clean_review_title),
        "images": record.get("images", []),
    }
    return normalized


def normalize_meta_record(
    record: Dict[str, Any],
    *,
    category: str | None = None,
    source_file: str | None = None,
) -> Dict[str, Any]:
    """Normalize an Amazon metadata record into a project-friendly schema."""
    normalized: Dict[str, Any] = {
        "product_id": record.get("parent_asin"),
        "parent_asin": record.get("parent_asin"),
        "category": category or record.get("main_category"),
        "dataset_version": "amazon_reviews_2023",
        "source_file": source_file,
        "main_category": record.get("main_category"),
        "product_title": record.get("title"),
        "average_rating": _safe_float(record.get("average_rating")),
        "rating_number": _safe_int(record.get("rating_number")),
        "features": record.get("features", []),
        "description": record.get("description", []),
        "price": _safe_float(record.get("price")),
        "images": record.get("images", []),
        "videos": record.get("videos", []),
        "store": record.get("store"),
        "categories": record.get("categories", []),
        "details": record.get("details", {}),
        "bought_together": record.get("bought_together"),
    }
    return normalized


def build_meta_index(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index metadata records by parent ASIN."""
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        parent_asin = record.get("parent_asin")
        if parent_asin:
            index[parent_asin] = record
    return index


def attach_metadata(
    reviews: Iterable[Dict[str, Any]],
    meta_index: Dict[str, Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """Attach normalized metadata to each normalized review record."""
    for review in reviews:
        parent_asin = review.get("parent_asin")
        enriched_review = dict(review)
        enriched_review["metadata"] = meta_index.get(parent_asin)
        yield enriched_review


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str | Path) -> int:
    """Write dictionaries to a JSONL file and return the record count."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def build_review_id(record: Dict[str, Any]) -> str:
    """Create a stable review identifier from available review fields."""
    user_id = record.get("user_id", "unknown_user")
    product_id = record.get("parent_asin") or record.get("asin") or "unknown_product"
    timestamp = (
        record.get("sort_timestamp")
        or record.get("timestamp")
        or "unknown_timestamp"
    )
    return f"{user_id}_{product_id}_{timestamp}"
