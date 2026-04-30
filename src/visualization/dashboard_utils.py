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


ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_DEMO_MODEL = "doubao-seed-2-0-lite-260215"

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
    "maybe",
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
    "summary": {"总结", "概括", "summary", "summarize", "brief", "overview", "evidence"},
    "product": {"商品", "产品", "product", "asin", "category", "类目"},
    "action": {"建议", "行动", "改进", "next", "action", "recommend", "suggest", "check"},
}
GREETING_TERMS = {"hi", "hello", "hey", "你好", "您好"}


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
        if title.lower() in {"no", "n/a", "na", "none", "null"}:
            title = "Untitled review"
        line = f"- {title}"
        if include_excerpt and excerpt:
            line += f": {excerpt}"
        lines.append(line)
    return "\n".join(lines)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """Return a count with a simple English singular/plural label."""
    label = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {label}"


def _evidence_note(snapshot: Dict[str, Any]) -> str:
    """Return an evidence warning based on negative-review support."""
    negative_count = snapshot["negative_reviews"]
    if negative_count == 0:
        return (
            "No calibrated negative reviews are available in this scope, so complaint "
            "analysis is not reliable here."
        )
    if negative_count == 1:
        return (
            "Evidence is limited to 1 calibrated negative review, so this should be "
            "treated as one customer case rather than a recurring complaint theme."
        )
    if negative_count < 3:
        return (
            f"Evidence is limited to {negative_count} calibrated negative reviews, so "
            "the result should be read as weak evidence rather than a stable trend."
        )
    return ""


def _format_available_negative_evidence(snapshot: Dict[str, Any], limit: int = 3) -> str:
    """Format the available negative-review evidence for sparse scopes."""
    representative = snapshot["representative_reviews"]
    if representative.empty:
        return "- No calibrated negative review evidence is available in this scope."
    return _format_review_bullets(representative.head(limit), include_excerpt=True)


def _count_answer(snapshot: Dict[str, Any], scope_label: str) -> str:
    """Answer count-style questions with explicit evidence boundaries."""
    note = _evidence_note(snapshot)
    answer = (
        f"The current scope is '{scope_label}'. It contains "
        f"{_plural(snapshot['total_reviews'], 'review')}, "
        f"{_plural(snapshot['calibrated_sentiment_reviews'], 'calibrated sentiment review')}, "
        f"{_plural(snapshot['negative_reviews'], 'calibrated negative review')}, and "
        f"{_plural(snapshot['high_value_negative_reviews'], 'high-value negative review')}."
    )
    return f"{answer}\n\nEvidence note: {note}" if note else answer


def _action_answer(snapshot: Dict[str, Any], scope_label: str) -> str:
    """Suggest merchant actions without overstating sparse evidence."""
    note = _evidence_note(snapshot)
    keyword_text = ", ".join(keyword for keyword, _ in snapshot["top_keywords"][:3])
    if snapshot["negative_reviews"] == 0:
        return (
            f"For '{scope_label}', I would not draw complaint conclusions yet because "
            "there are no calibrated negative reviews in the current scope.\n\n"
            "Practical next steps: broaden the scope to the category, inspect uploaded "
            "merchant reviews if available, or use Single Review Check for new customer feedback."
        )
    if snapshot["negative_reviews"] < 3:
        issue_hint = keyword_text or "the available negative review text"
        return (
            f"For '{scope_label}', use the available negative review as a case-level signal, "
            "not as proof of a recurring product issue.\n\n"
            f"Evidence note: {note}\n\n"
            f"Suggested next checks: read the full negative review, verify whether '{issue_hint}' "
            "appears in more reviews at category level, and check whether the product page, "
            "instructions, or support content can address that specific customer pain point."
        )

    keyword_text = keyword_text or "the top complaint keywords"
    return (
        f"For '{scope_label}', the negative-review evidence is large enough for a broader "
        f"scan. Start with the recurring terms ({keyword_text}), inspect representative "
        "reviews, and prioritise fixes that appear across multiple reviews."
    )


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
    """Convert dashboard context into a compact prompt for Ark LLM use."""
    keyword_text = ", ".join(keyword for keyword, _ in snapshot["top_keywords"][:6]) or "none"
    review_bullets = _format_review_bullets(snapshot["representative_reviews"])
    evidence_note = _evidence_note(snapshot) or "There is enough negative-review evidence for theme-level discussion."
    return (
        "You are an intelligent review analytics assistant for an e-commerce merchant. "
        "Use only the supplied dashboard context; do not invent products, counts, or reviews. "
        "Answer in the same language as the user's question. Follow this evidence policy: "
        "when there are fewer than 3 negative reviews, do not call the result a recurring "
        "theme or broad product problem. Frame it as available evidence or a case-level "
        "signal. If the user asks for more examples than exist, say how many are available. "
        "If the user asks for advice, give concise checks that match the evidence level.\n\n"
        f"Scope: {snapshot['scope_label']}\n"
        f"Total reviews: {snapshot['total_reviews']}\n"
        f"High-value reviews: {snapshot['high_value_reviews']}\n"
        f"Calibrated sentiment reviews: {snapshot['calibrated_sentiment_reviews']}\n"
        f"Negative reviews: {snapshot['negative_reviews']}\n"
        f"High-value negative reviews: {snapshot['high_value_negative_reviews']}\n"
        f"Evidence note: {evidence_note}\n"
        f"Average rating: {snapshot['average_rating']:.2f}\n"
        f"Unique products: {snapshot['unique_products']}\n"
        f"Top complaint keywords: {keyword_text}\n"
        "Representative reviews:\n"
        f"{review_bullets or '- No representative reviews available.'}\n\n"
        "When useful, structure the answer with: key finding, evidence, recommended action.\n"
        f"User question: {question}"
    )


def _answer_with_ark(snapshot: Dict[str, Any], question: str) -> str | None:
    """Use Volcengine Ark through the OpenAI-compatible Responses API."""
    api_key = os.getenv("ARK_API_KEY")
    model_name = os.getenv("ARK_MODEL", ARK_DEMO_MODEL)
    base_url = os.getenv("ARK_BASE_URL", ARK_BASE_URL)
    if not api_key or not model_name or OpenAI is None:
        return None

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=30.0,
        )
        response = client.responses.create(
            model=model_name,
            input=_build_context_prompt(snapshot, question),
            extra_body={
                "thinking": {"type": "disabled"},
            },
        )
        return (response.output_text or "").strip() or None
    except Exception:
        return None


def answer_chat_question(
    question: str,
    dataframe: pd.DataFrame,
    *,
    scope_label: str,
    use_ark_llm: bool = False,
) -> str:
    """Answer a chat question with Ark LLM and local-rule fallback."""
    snapshot = build_scope_snapshot(dataframe, scope_label=scope_label)

    if snapshot["total_reviews"] == 0:
        return "There is no data in the current scope yet. Try switching the category or product."

    lowered = question.lower()
    compact_question = re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered).strip()
    top_keywords = [keyword for keyword, _ in snapshot["top_keywords"]]
    representative = snapshot["representative_reviews"]
    evidence_note = _evidence_note(snapshot)

    if compact_question in GREETING_TERMS:
        if evidence_note:
            return (
                f"Hi. You are viewing '{scope_label}'. This scope has "
                f"{_plural(snapshot['total_reviews'], 'review')} and "
                f"{_plural(snapshot['negative_reviews'], 'calibrated negative review')}.\n\n"
                f"{evidence_note}\n\n"
                "I can help inspect available evidence, explain counts, or suggest what to check next."
            )
        return (
            f"Hi. You are viewing '{scope_label}'. I can help summarize complaint themes, "
            "show representative negative reviews, or suggest merchant actions."
        )

    if any(keyword in lowered for keyword in QUESTION_KEYWORDS["count"]):
        return _count_answer(snapshot, scope_label)

    if any(keyword in lowered for keyword in QUESTION_KEYWORDS["problem"]):
        if snapshot["negative_reviews"] == 0:
            return (
                f"I cannot identify complaint themes for '{scope_label}' because there are "
                "no calibrated negative reviews in this scope. Try broadening to the category "
                "or checking uploaded merchant reviews."
            )
        if snapshot["negative_reviews"] < 3:
            keyword_text = ", ".join(top_keywords[:3]) if top_keywords else "no clear repeated terms"
            return (
                f"For '{scope_label}', I would not call these 'main themes' yet. "
                f"{evidence_note}\n\n"
                f"Available issue signal: {keyword_text}.\n"
                f"Evidence:\n{_format_available_negative_evidence(snapshot, limit=2)}"
            )
        keyword_text = ", ".join(top_keywords[:5]) if top_keywords else "no clear recurring complaint terms yet"
        bullet_text = _format_review_bullets(representative.head(2), include_excerpt=False)
        return (
            f"In '{scope_label}', the main complaint themes are: {keyword_text}."
            f"\nRepresentative negative reviews:\n{bullet_text or '- No representative negative reviews available yet.'}"
        )

    if any(keyword in lowered for keyword in QUESTION_KEYWORDS["example"]):
        if snapshot["negative_reviews"] == 0:
            return "There are no calibrated negative review examples in the current scope."
        if snapshot["negative_reviews"] < 3:
            verb = "is" if snapshot["negative_reviews"] == 1 else "are"
            return (
                f"Only {_plural(snapshot['negative_reviews'], 'calibrated negative review')} {verb} "
                f"available in '{scope_label}', so I can show the available evidence rather "
                f"than multiple representative examples:\n"
                f"{_format_available_negative_evidence(snapshot, limit=3)}"
            )
        review_examples = _format_review_bullets(representative, include_excerpt=True)
        return review_examples or "There are no representative review examples in the current scope yet."

    if any(keyword in lowered for keyword in QUESTION_KEYWORDS["summary"]):
        keyword_text = ", ".join(top_keywords[:5]) if top_keywords else "no clear complaint keywords"
        if evidence_note:
            evidence_text = ""
            if snapshot["negative_reviews"] > 0:
                evidence_text = f"\n\nEvidence:\n{_format_available_negative_evidence(snapshot, limit=2)}"
            return (
                f"At a glance, '{scope_label}' has {snapshot['total_reviews']} reviews, "
                f"an average rating of {snapshot['average_rating']:.2f}, and "
                f"{_plural(snapshot['negative_reviews'], 'calibrated negative review')}.\n\n"
                f"{evidence_note}\n\n"
                f"Available negative signal: {keyword_text}."
                f"{evidence_text}"
            )
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

    if any(keyword in lowered for keyword in QUESTION_KEYWORDS["action"]):
        return _action_answer(snapshot, scope_label)

    if use_ark_llm:
        llm_answer = _answer_with_ark(snapshot, question)
        if llm_answer:
            return llm_answer

    retrieved = _retrieve_reviews_for_question(question, dataframe[dataframe["sentiment_label"] == "negative"])
    if not retrieved.empty:
        if evidence_note:
            return (
                f"{evidence_note}\n\n"
                f"Available negative review evidence in '{scope_label}':\n"
                f"{_format_review_bullets(retrieved, include_excerpt=True)}"
            )
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
        "Summarize available negative evidence",
        "Show available negative reviews",
        "What should the merchant check next?",
    ]
