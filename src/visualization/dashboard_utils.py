"""Utilities for preparing dashboard-ready data, charts, and chat context."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


NUMERIC_COLUMNS = {
    "rating",
    "helpful_votes",
    "review_value_label",
    "review_text_word_count",
    "review_text_char_count",
    "sentiment_target",
    "lex_score",
}
BOOL_COLUMNS = {"verified_purchase"}
STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "don",
    "don't",
    "even",
    "for",
    "from",
    "get",
    "got",
    "had",
    "has",
    "have",
    "here",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "it's",
    "like",
    "look",
    "made",
    "make",
    "makes",
    "my",
    "no",
    "not",
    "of",
    "off",
    "on",
    "one",
    "or",
    "out",
    "product",
    "really",
    "right",
    "so",
    "some",
    "star",
    "stars",
    "still",
    "such",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "too",
    "tried",
    "two",
    "use",
    "used",
    "using",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "with",
    "would",
    "you",
    "your",
}
QUESTION_KEYWORDS = {
    "count": {"多少", "几条", "数量", "count", "how many", "总数"},
    "problem": {"问题", "抱怨", "主要", "痛点", "complaint", "issue", "problem"},
    "example": {"例子", "举例", "评论", "example", "review", "sample"},
    "summary": {"总结", "概括", "summary", "brief", "overview"},
    "product": {"商品", "产品", "product", "asin", "category", "类目"},
}


def _read_split_csvs(prefix: str, processed_dir: Path) -> pd.DataFrame:
    """Read train, validation, and test CSV files for one dataset prefix."""
    frames: List[pd.DataFrame] = []
    for split in ("train", "validation", "test"):
        file_path = processed_dir / f"{prefix}_{split}.csv"
        frame = pd.read_csv(file_path)
        frame["split"] = frame["split"].fillna(split) if "split" in frame.columns else split
        frames.append(frame)

    dataframe = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS.intersection(dataframe.columns):
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    for column in BOOL_COLUMNS.intersection(dataframe.columns):
        dataframe[column] = dataframe[column].astype(str).str.lower().map(
            {"true": True, "false": False}
        )
    return dataframe


def _top_keywords(texts: List[str], *, limit: int = 8) -> List[Tuple[str, int]]:
    """Return simple frequency-based complaint keywords."""
    counts: Dict[str, int] = {}
    for text in texts:
        for token in re.findall(r"[a-zA-Z']+", str(text).lower()):
            if len(token) < 3 or token in STOPWORDS:
                continue
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:limit]


def _select_representative_reviews(dataframe: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Pick representative reviews for display or chat context."""
    if dataframe.empty:
        return dataframe

    ranked = dataframe.copy()
    ranked["review_value_label"] = ranked["review_value_label"].fillna(0)
    ranked["helpful_votes"] = ranked["helpful_votes"].fillna(0)
    ranked["review_text_word_count"] = ranked["review_text_word_count"].fillna(0)
    ranked["rating"] = ranked["rating"].fillna(0)
    ranked = ranked.sort_values(
        by=["review_value_label", "helpful_votes", "review_text_word_count", "rating"],
        ascending=[False, False, False, True],
    )
    return ranked.head(limit)


def filter_scope_dataframe(
    merged_reviews: pd.DataFrame,
    *,
    category: str | None = None,
    product_id: str | None = None,
    split: str = "all",
) -> pd.DataFrame:
    """Filter the merged review frame by category, product, and split."""
    filtered = merged_reviews.copy()
    if split != "all":
        filtered = filtered[filtered["split"] == split]
    if category and category != "All":
        filtered = filtered[filtered["category"] == category]
    if product_id and product_id != "All":
        filtered = filtered[filtered["product_id"] == product_id]
    return filtered.reset_index(drop=True)


def build_scope_snapshot(dataframe: pd.DataFrame, *, scope_label: str) -> Dict[str, Any]:
    """Create summary statistics for the current dashboard scope."""
    sentiment_rows = dataframe[dataframe["sentiment_label"] != "unlabeled"]
    negative_rows = sentiment_rows[sentiment_rows["sentiment_label"] == "negative"]
    high_value_negative = negative_rows[negative_rows["review_value_label"] == 1]
    top_keywords = _top_keywords(negative_rows["clean_review_text"].fillna("").tolist())
    representative = _select_representative_reviews(
        high_value_negative if not high_value_negative.empty else negative_rows,
        limit=3,
    )

    return {
        "scope_label": scope_label,
        "total_reviews": int(len(dataframe)),
        "high_value_reviews": int((dataframe["review_value_label"] == 1).sum()),
        "calibrated_sentiment_reviews": int(len(sentiment_rows)),
        "negative_reviews": int(len(negative_rows)),
        "high_value_negative_reviews": int(len(high_value_negative)),
        "verified_purchase_rate": float(
            dataframe["verified_purchase"].fillna(False).mean() if len(dataframe) else 0.0
        ),
        "average_rating": float(dataframe["rating"].mean()) if len(dataframe) else 0.0,
        "unique_products": int(dataframe["product_id"].nunique()) if "product_id" in dataframe else 0,
        "top_keywords": top_keywords,
        "representative_reviews": representative,
    }


def build_dashboard_payload(processed_dir: Path | str = Path("data/processed")) -> Dict[str, Any]:
    """Prepare merged outputs and overview statistics for the dashboard."""
    processed_dir = Path(processed_dir)
    review_value_df = _read_split_csvs("review_value", processed_dir)
    sentiment_df = _read_split_csvs("sentiment", processed_dir)

    sentiment_subset = sentiment_df[
        ["review_id", "sentiment_label", "sentiment_target", "lex_score"]
    ].drop_duplicates("review_id")
    merged_reviews = review_value_df.merge(
        sentiment_subset,
        on="review_id",
        how="left",
    )
    merged_reviews["sentiment_label"] = merged_reviews["sentiment_label"].fillna("unlabeled")
    merged_reviews["sentiment_target"] = merged_reviews["sentiment_target"].fillna(-1)
    merged_reviews["lex_score"] = merged_reviews["lex_score"].fillna(0.0)

    category_options = ["All"] + sorted(merged_reviews["category"].dropna().unique().tolist())
    products_by_category: Dict[str, List[Tuple[str, str]]] = {"All": [("All", "All products")]}
    for category in sorted(merged_reviews["category"].dropna().unique()):
        product_frame = (
            merged_reviews[merged_reviews["category"] == category][
                ["product_id", "product_title"]
            ]
            .drop_duplicates()
            .sort_values(by=["product_title", "product_id"])
        )
        products_by_category[category] = [("All", "All products")] + list(
            product_frame.itertuples(index=False, name=None)
        )

    overview_snapshot = build_scope_snapshot(merged_reviews, scope_label="Entire evaluation set")
    return {
        "review_value": review_value_df,
        "sentiment": sentiment_df,
        "merged_reviews": merged_reviews,
        "category_options": category_options,
        "products_by_category": products_by_category,
        "overview_snapshot": overview_snapshot,
    }


def build_category_overview_figure(merged_reviews: pd.DataFrame):
    """Build a category-level stacked bar chart."""
    frame = (
        merged_reviews.assign(
            sentiment_bucket=merged_reviews["sentiment_label"].replace({"unlabeled": "other"})
        )
        .groupby(["category", "sentiment_bucket"], dropna=False)
        .size()
        .reset_index(name="review_count")
    )
    return px.bar(
        frame,
        x="category",
        y="review_count",
        color="sentiment_bucket",
        title="Review Distribution by Category",
        color_discrete_map={
            "positive": "#4b7f52",
            "negative": "#b44c43",
            "other": "#d6c6a5",
        },
    )


def build_keyword_figure(top_keywords: List[Tuple[str, int]]):
    """Build a keyword bar chart from frequency tuples."""
    if not top_keywords:
        return None
    frame = pd.DataFrame(top_keywords, columns=["keyword", "count"])
    return px.bar(
        frame,
        x="count",
        y="keyword",
        orientation="h",
        title="Top Complaint Keywords",
        color="count",
        color_continuous_scale=["#d6c6a5", "#d37a51", "#b44c43"],
    )


def _format_review_bullets(dataframe: pd.DataFrame, *, include_excerpt: bool = True) -> str:
    """Format representative reviews into compact bullets."""
    lines: List[str] = []
    for _, row in dataframe.iterrows():
        excerpt = str(row.get("clean_review_text", "")).strip()
        if len(excerpt) > 140:
            excerpt = excerpt[:137] + "..."
        title = str(row.get("clean_review_title", "")).strip() or "Untitled review"
        line = f"- {title}"
        if include_excerpt and excerpt:
            line += f": {excerpt}"
        lines.append(line)
    return "\n".join(lines)


def _retrieve_reviews_for_question(question: str, dataframe: pd.DataFrame, limit: int = 2) -> pd.DataFrame:
    """Find reviews with the highest token overlap to the question."""
    query_terms = {
        token for token in re.findall(r"[a-zA-Z']+", question.lower()) if len(token) >= 3
    }
    if not query_terms or dataframe.empty:
        return dataframe.head(limit)

    scored_rows = []
    for _, row in dataframe.iterrows():
        text = f"{row.get('clean_review_title', '')} {row.get('clean_review_text', '')}".lower()
        score = sum(1 for token in query_terms if token in text)
        scored_rows.append((score, row))

    scored_rows.sort(
        key=lambda item: (
            item[0],
            item[1].get("review_value_label", 0),
            item[1].get("helpful_votes", 0),
        ),
        reverse=True,
    )
    selected_rows = [item[1] for item in scored_rows if item[0] > 0][:limit]
    if not selected_rows:
        return _select_representative_reviews(dataframe, limit=limit)
    return pd.DataFrame(selected_rows)


def _build_context_prompt(snapshot: Dict[str, Any], question: str) -> str:
    """Convert dashboard context into a compact prompt for optional LLM use."""
    keyword_text = ", ".join(keyword for keyword, _ in snapshot["top_keywords"][:6]) or "none"
    review_bullets = _format_review_bullets(snapshot["representative_reviews"])
    return (
        "You are a review analytics assistant for an e-commerce merchant. "
        "Answer in concise English based only on the supplied dashboard context.\n\n"
        f"Scope: {snapshot['scope_label']}\n"
        f"Total reviews: {snapshot['total_reviews']}\n"
        f"High-value reviews: {snapshot['high_value_reviews']}\n"
        f"Calibrated sentiment reviews: {snapshot['calibrated_sentiment_reviews']}\n"
        f"Negative reviews: {snapshot['negative_reviews']}\n"
        f"High-value negative reviews: {snapshot['high_value_negative_reviews']}\n"
        f"Average rating: {snapshot['average_rating']:.2f}\n"
        f"Top complaint keywords: {keyword_text}\n"
        "Representative reviews:\n"
        f"{review_bullets or '- No representative reviews available.'}\n\n"
        f"User question: {question}"
    )


def _answer_with_openai(snapshot: Dict[str, Any], question: str) -> str | None:
    """Use the OpenAI API if the environment is configured for it."""
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL")
    if not api_key or not model_name or OpenAI is None:
        return None

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model_name,
            input=_build_context_prompt(snapshot, question),
        )
        return (response.output_text or "").strip() or None
    except Exception:
        return None


def answer_chat_question(
    question: str,
    dataframe: pd.DataFrame,
    *,
    scope_label: str,
    use_external_llm: bool = False,
) -> str:
    """Answer a chat question with local rules and optional LLM fallback."""
    snapshot = build_scope_snapshot(dataframe, scope_label=scope_label)
    if use_external_llm:
        llm_answer = _answer_with_openai(snapshot, question)
        if llm_answer:
            return llm_answer

    if snapshot["total_reviews"] == 0:
        return "There is no data in the current scope yet. Try switching the category or product."

    lowered = question.lower()
    top_keywords = [keyword for keyword, _ in snapshot["top_keywords"]]
    representative = snapshot["representative_reviews"]

    if any(keyword in lowered for keyword in QUESTION_KEYWORDS["count"]):
        return (
            f"The current scope is '{scope_label}'. It contains {snapshot['total_reviews']} reviews, "
            f"including {snapshot['high_value_reviews']} high-value reviews, "
            f"{snapshot['calibrated_sentiment_reviews']} sentiment-labeled reviews, "
            f"{snapshot['negative_reviews']} negative reviews, and "
            f"{snapshot['high_value_negative_reviews']} high-value negative reviews."
        )

    if any(keyword in lowered for keyword in QUESTION_KEYWORDS["problem"]):
        keyword_text = ", ".join(top_keywords[:5]) if top_keywords else "no clear recurring complaint terms yet"
        bullet_text = _format_review_bullets(representative.head(2), include_excerpt=False)
        return (
            f"In '{scope_label}', the main complaint themes are: {keyword_text}."
            f"\nRepresentative negative reviews:\n{bullet_text or '- No representative negative reviews available yet.'}"
        )

    if any(keyword in lowered for keyword in QUESTION_KEYWORDS["example"]):
        review_examples = _format_review_bullets(representative, include_excerpt=True)
        return review_examples or "There are no representative review examples in the current scope yet."

    if any(keyword in lowered for keyword in QUESTION_KEYWORDS["summary"]):
        keyword_text = ", ".join(top_keywords[:5]) if top_keywords else "no clear complaint keywords"
        return (
            f"At a glance, '{scope_label}' has an average rating of {snapshot['average_rating']:.2f}, "
            f"a high-value review share of "
            f"{snapshot['high_value_reviews'] / max(snapshot['total_reviews'], 1):.0%}, "
            f"and its main negative themes revolve around {keyword_text}."
        )

    if any(keyword in lowered for keyword in QUESTION_KEYWORDS["product"]):
        return (
            f"You are currently viewing '{scope_label}'. This scope contains "
            f"{snapshot['unique_products']} products with an average rating of "
            f"{snapshot['average_rating']:.2f}."
        )

    retrieved = _retrieve_reviews_for_question(question, dataframe[dataframe["sentiment_label"] == "negative"])
    if not retrieved.empty:
        return (
            f"I found the following negative reviews that are most relevant to your question in '{scope_label}':\n"
            f"{_format_review_bullets(retrieved, include_excerpt=True)}"
        )

    keyword_text = ", ".join(top_keywords[:5]) if top_keywords else "no clear complaint keywords"
    return (
        f"For '{scope_label}', the most reliable summary I can give right now is: "
        f"{snapshot['total_reviews']} total reviews, {snapshot['negative_reviews']} negative reviews, "
        f"and the main complaint keywords include {keyword_text}."
    )


def get_chat_suggestions(scope_label: str) -> List[str]:
    """Return a short list of chat prompt suggestions."""
    return [
        f"How many high-value negative reviews are in {scope_label}?",
        f"What are the main complaint themes in {scope_label}?",
        "Show me two representative negative review examples.",
    ]
