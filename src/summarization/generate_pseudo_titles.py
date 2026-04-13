"""Generate cleaner complaint-title pseudo labels with Volcengine Ark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_DEMO_MODEL = "doubao-seed-2-0-lite-260215"
INVALID_TITLE_MARKERS = {
    "one star",
    "two stars",
    "three stars",
    "four stars",
    "five stars",
    "1 star",
    "2 stars",
    "3 stars",
    "4 stars",
    "5 stars",
    "customer review",
    "review title",
    "complaint title",
}
GENERIC_TITLES = {
    "bad product",
    "not good",
    "poor quality",
    "disappointed",
    "worthless",
    "not worth it",
    "waste of money",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Use Ark as a teacher model to generate normalized complaint-title "
            "pseudo labels for negative reviews."
        )
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/sentiment_train.csv"),
        help="Path to the calibrated sentiment training CSV file.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/processed/sentiment_validation.csv"),
        help="Path to the calibrated sentiment validation CSV file.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/processed/sentiment_test.csv"),
        help="Path to the calibrated sentiment test CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where pseudo-summary CSV files will be written.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/processed/pseudo_summary_manifest.json"),
        help="Path to save generation metadata and quality-filter counts.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="clean_review_text",
        help="Review text column used as model input.",
    )
    parser.add_argument(
        "--original-title-column",
        type=str,
        default="clean_review_title",
        help="Original noisy review-title column kept for comparison.",
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
        "--model",
        type=str,
        default=os.getenv("ARK_MODEL", ARK_DEMO_MODEL),
        help="Ark text-generation model ID. Defaults to ARK_MODEL or a demo model.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("ARK_BASE_URL", ARK_BASE_URL),
        help="Ark OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of reviews to send in each Ark request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional per-split row limit for quick smoke tests.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Pause between API requests to avoid rate-limit spikes.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="How many times to retry a failed Ark request.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Per-request timeout (seconds) for each Ark API call.",
    )
    return parser.parse_args()


def load_local_env(env_path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from a local .env file without overriding exports."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_negative_rows(
    file_path: Path,
    *,
    text_column: str,
    original_title_column: str,
    sentiment_column: str,
    negative_label: str,
    limit: int | None,
) -> List[Dict[str, Any]]:
    """Load negative rows while preserving source columns."""
    rows: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if (row.get(sentiment_column) or "").strip() != negative_label:
                continue

            review_text = (row.get(text_column) or "").strip()
            if not review_text:
                continue

            enriched = dict(row)
            enriched["original_review_title"] = (
                row.get(original_title_column) or ""
            ).strip()
            rows.append(enriched)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_generation_prompt(rows: List[Dict[str, Any]], *, text_column: str) -> str:
    """Build a strict JSON-generation prompt for one batch."""
    items = []
    for index, row in enumerate(rows, start=1):
        review_text = str(row.get(text_column, "")).strip()
        if len(review_text) > 900:
            review_text = review_text[:897] + "..."
        items.append({"id": index, "review_text": review_text})

    return (
        "You are labeling negative e-commerce reviews for a complaint-title model.\n"
        "For each review, write one short English complaint title.\n"
        "Rules:\n"
        "- Use 3 to 8 words.\n"
        "- Focus on the concrete product or service problem.\n"
        "- Use only facts stated in the review text.\n"
        "- Do not mention star ratings.\n"
        "- Do not include generic titles like 'bad product' unless the review has no detail.\n"
        "- Return valid JSON only, with this exact shape: "
        "[{\"id\": 1, \"title\": \"...\"}]\n\n"
        f"Reviews:\n{json.dumps(items, ensure_ascii=False)}"
    )


def strip_json_markdown(text: str) -> str:
    """Remove common JSON markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def parse_titles(response_text: str, expected_count: int) -> List[str]:
    """Parse Ark JSON output into title strings."""
    data = json.loads(strip_json_markdown(response_text))
    if not isinstance(data, list):
        raise ValueError("Ark response was not a JSON list.")

    titles_by_id: Dict[int, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        title = str(item.get("title", "")).strip()
        titles_by_id[item_id] = title

    titles = [titles_by_id.get(index, "") for index in range(1, expected_count + 1)]
    if any(not title for title in titles):
        raise ValueError("Ark response did not include every requested title.")
    return titles


def normalize_title(title: str) -> str:
    """Normalize a generated title for training."""
    title = re.sub(r"""^[-"'`*_: ]+|[-"'`*_: .!]+$""", "", title.strip())
    title = re.sub(r"\s+", " ", title)
    words = title.split()
    if len(words) > 10:
        title = " ".join(words[:10])
    return title


def fallback_title(review_text: str) -> str:
    """Build a deterministic title when the teacher output is unusable."""
    clauses = re.split(r"[.!?;]| but | and | because | while | although ", review_text, maxsplit=5)
    cues = (
        "not",
        "never",
        "no ",
        "broke",
        "broken",
        "stopped",
        "waste",
        "poor",
        "cheap",
        "wrong",
        "issue",
        "problem",
        "smell",
        "hard",
        "too",
        "loose",
        "scratch",
        "crack",
        "leak",
        "missing",
        "returned",
    )
    chosen = review_text.strip()
    for clause in clauses:
        candidate = clause.strip()
        if candidate and any(cue in candidate.lower() for cue in cues):
            chosen = candidate
            break

    words = chosen.split()
    if len(words) > 8:
        chosen = " ".join(words[:8])
    return normalize_title(chosen).capitalize()


def is_low_quality_title(title: str, review_text: str) -> bool:
    """Return True when a generated title is too generic or malformed."""
    normalized = normalize_title(title)
    lowered = normalized.lower()
    words = normalized.split()
    if len(words) < 2 or len(words) > 10:
        return True
    if any(marker in lowered for marker in INVALID_TITLE_MARKERS):
        return True
    if lowered in GENERIC_TITLES and len(review_text.split()) > 8:
        return True
    return False


def call_ark_titles(
    rows: List[Dict[str, Any]],
    *,
    client: OpenAI,
    model: str,
    text_column: str,
    max_retries: int,
) -> List[str]:
    """Generate one batch of pseudo titles with retries."""
    prompt = build_generation_prompt(rows, text_column=text_column)
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return parse_titles(response.output_text or "", expected_count=len(rows))
        except Exception as exc:  # noqa: BLE001 - logged into manifest and retried.
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Ark title generation failed: {last_error}") from last_error


def enrich_rows_with_titles(
    rows: List[Dict[str, Any]],
    *,
    client: OpenAI,
    model: str,
    text_column: str,
    batch_size: int,
    sleep_seconds: float,
    max_retries: int,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Generate, clean, and validate pseudo titles for a split."""
    enriched_rows: List[Dict[str, Any]] = []
    stats = {
        "input_rows": len(rows),
        "ark_titles": 0,
        "fallback_titles": 0,
        "failed_batches": 0,
    }

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_source = "ark"
        try:
            titles = call_ark_titles(
                batch,
                client=client,
                model=model,
                text_column=text_column,
                max_retries=max_retries,
            )
        except RuntimeError:
            stats["failed_batches"] += 1
            titles = [fallback_title(str(row.get(text_column, ""))) for row in batch]
            batch_source = "fallback"

        for row, title in zip(batch, titles):
            review_text = str(row.get(text_column, "")).strip()
            cleaned_title = normalize_title(title)
            source = batch_source
            if is_low_quality_title(cleaned_title, review_text):
                cleaned_title = fallback_title(review_text)
                source = "fallback"

            updated = dict(row)
            updated["llm_complaint_title"] = cleaned_title
            updated["pseudo_title_source"] = source
            enriched_rows.append(updated)
            stats["ark_titles" if source == "ark" else "fallback_titles"] += 1

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return enriched_rows, stats


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_pseudo_titles(args: argparse.Namespace) -> Dict[str, Any]:
    """Generate pseudo-summary datasets for train, validation, and test splits."""
    if OpenAI is None:
        raise RuntimeError("The openai package is required for Ark-compatible calls.")

    load_local_env()
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise RuntimeError("Set ARK_API_KEY or create a local .env file before running.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.request_timeout,
    )

    split_files = {
        "train": args.train_file,
        "validation": args.validation_file,
        "test": args.test_file,
    }
    manifest: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "base_url": args.base_url,
            "text_column": args.text_column,
            "original_title_column": args.original_title_column,
            "sentiment_column": args.sentiment_column,
            "negative_label": args.negative_label,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "request_timeout": args.request_timeout,
            "quality_rules": {
                "target_words": "3 to 8 preferred; 2 to 10 accepted after cleanup",
                "rejects_star_rating_titles": True,
                "rejects_generic_titles": True,
                "fallback": "short clause extracted from the review text",
            },
        },
        "splits": {},
        "files": {},
    }

    for split, file_path in split_files.items():
        rows = load_negative_rows(
            file_path,
            text_column=args.text_column,
            original_title_column=args.original_title_column,
            sentiment_column=args.sentiment_column,
            negative_label=args.negative_label,
            limit=args.limit,
        )
        enriched_rows, stats = enrich_rows_with_titles(
            rows,
            client=client,
            model=args.model,
            text_column=args.text_column,
            batch_size=args.batch_size,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )
        output_path = args.output_dir / f"pseudo_summary_{split}.csv"
        write_csv(enriched_rows, output_path)
        manifest["splits"][split] = stats
        manifest["files"][split] = str(output_path)
        print(
            f"{split}: wrote {len(enriched_rows)} rows "
            f"({stats['ark_titles']} Ark / {stats['fallback_titles']} fallback)"
        )

    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest written to {args.manifest_output}")
    return manifest


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    manifest = generate_pseudo_titles(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
