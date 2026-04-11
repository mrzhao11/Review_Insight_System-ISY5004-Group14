"""Streamlit application entry point for the review insight system."""

from __future__ import annotations

from html import escape
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, List

import joblib
import pandas as pd
import streamlit as st
import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.clean_text import clean_text
from src.visualization.dashboard_utils import (
    answer_chat_question,
    build_category_overview_figure,
    build_dashboard_payload,
    build_keyword_figure,
    build_scope_snapshot,
    filter_scope_dataframe,
    get_chat_suggestions,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BERT_MODEL_DIR = MODELS_DIR / "bert_sentiment"
SUMMARY_MODEL_NAME = "google/flan-t5-small"


def inject_styles() -> None:
    """Apply lightweight custom styling."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(214, 198, 165, 0.28), transparent 32%),
                linear-gradient(180deg, #f7f3ea 0%, #f1ebe0 100%);
            color: #1f1e1a;
            font-family: "Trebuchet MS", "Avenir Next", sans-serif;
        }
        .hero-card {
            padding: 1.4rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #fdf7eb 0%, #efe1c7 100%);
            border: 1px solid rgba(124, 90, 54, 0.12);
            box-shadow: 0 18px 40px rgba(92, 66, 37, 0.08);
            margin-bottom: 1rem;
        }
        .hero-eyebrow {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.78rem;
            color: #9a5a38;
            margin-bottom: 0.2rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            color: #2b251f;
            margin-bottom: 0.4rem;
        }
        .hero-copy {
            color: #5a5147;
            line-height: 1.55;
            margin-bottom: 0;
        }
        .section-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(255, 251, 244, 0.92);
            border: 1px solid rgba(124, 90, 54, 0.12);
            box-shadow: 0 12px 24px rgba(92, 66, 37, 0.06);
        }
        .review-card {
            border-left: 4px solid #b44c43;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            background: rgba(255, 248, 245, 0.94);
            margin-bottom: 0.8rem;
        }
        .review-title {
            font-weight: 700;
            color: #3a2d25;
            margin-bottom: 0.25rem;
        }
        .review-meta {
            font-size: 0.86rem;
            color: #7b6c5e;
            margin-bottom: 0.35rem;
        }
        .review-body {
            color: #4c4339;
            line-height: 1.55;
        }
        .chat-row {
            display: flex;
            margin-bottom: 0.85rem;
        }
        .chat-row.user {
            justify-content: flex-end;
        }
        .chat-row.assistant {
            justify-content: flex-start;
        }
        .chat-bubble {
            max-width: 78%;
            padding: 0.9rem 1rem;
            border-radius: 18px;
            box-shadow: 0 8px 18px rgba(46, 34, 23, 0.08);
            line-height: 1.55;
            font-size: 0.96rem;
            white-space: normal;
            word-break: break-word;
        }
        .chat-bubble.user {
            background: #2f2b26;
            color: #f9f5ee;
            border-bottom-right-radius: 8px;
        }
        .chat-bubble.assistant {
            background: rgba(255, 251, 244, 0.95);
            color: #312a23;
            border: 1px solid rgba(124, 90, 54, 0.10);
            border-bottom-left-radius: 8px;
        }
        .chat-role {
            font-size: 0.74rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            opacity: 0.72;
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_payload() -> dict[str, Any]:
    """Load processed dashboard data."""
    return build_dashboard_payload(PROCESSED_DIR)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    """Load saved model metrics for dashboard display."""
    return {
        "review_value": json.loads((MODELS_DIR / "review_value_metrics.json").read_text()),
        "sentiment": json.loads((MODELS_DIR / "bert_sentiment_metrics.json").read_text()),
        "summary": json.loads((MODELS_DIR / "t5_summary_metrics.json").read_text()),
        "summary_samples": json.loads((MODELS_DIR / "t5_summary_samples.json").read_text()),
    }


@st.cache_resource(show_spinner=False)
def load_review_value_bundle() -> dict[str, Any]:
    """Load the trained review-value classifier bundle."""
    bundle = joblib.load(MODELS_DIR / "review_value_classifier.pkl")
    return bundle if isinstance(bundle, dict) else {"pipeline": bundle, "decision_threshold": 0.5}


@st.cache_resource(show_spinner=False)
def load_sentiment_bundle() -> tuple[BertTokenizer, BertForSequenceClassification, torch.device]:
    """Load the fine-tuned BERT sentiment classifier."""
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR, local_files_only=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner=False)
def load_summary_bundle() -> tuple[T5Tokenizer, T5ForConditionalGeneration, torch.device]:
    """Load the cached zero-shot Flan-T5 checkpoint used for demo summarization."""
    tokenizer = T5Tokenizer.from_pretrained(SUMMARY_MODEL_NAME, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(SUMMARY_MODEL_NAME, local_files_only=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_review_value(text: str) -> dict[str, float]:
    """Predict the review-value label for a text snippet."""
    bundle = load_review_value_bundle()
    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("decision_threshold", 0.5))
    probability = float(pipeline.predict_proba([text])[0][1])
    return {
        "label": int(probability >= threshold),
        "probability": probability,
        "threshold": threshold,
    }


def predict_sentiment(text: str) -> dict[str, float | str]:
    """Predict sentiment with the fine-tuned BERT model."""
    tokenizer, model, device = load_sentiment_bundle()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    label_index = int(torch.argmax(logits, dim=-1).item())
    label_name = model.config.id2label.get(label_index, str(label_index)).lower()
    return {
        "label": label_name,
        "negative_probability": float(probabilities[0]),
        "positive_probability": float(probabilities[1]),
    }


def summarize_issue(text: str) -> str:
    """Generate a short complaint title with the pretrained T5 model."""
    tokenizer, model, device = load_summary_bundle()
    prompt = (
        "Write a short complaint title for this customer review. "
        "Keep it concise and focused on the main problem.\n\n"
        f"Review: {text.strip()}"
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_length=24,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return postprocess_summary(generated, text)


def build_fallback_title(text: str) -> str:
    """Create a compact fallback issue title when zero-shot output is noisy."""
    normalized = clean_text(text)
    clauses = re.split(r"[.!?;]| but | and | because | while | although ", normalized, maxsplit=4)
    negative_cues = (
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
        "didn't",
        "doesn't",
        "won't",
        "wouldn't",
        "loose",
        "scratch",
        "crack",
    )
    chosen = normalized
    for clause in clauses:
        candidate = clause.strip()
        lowered = candidate.lower()
        if candidate and any(cue in lowered for cue in negative_cues):
            chosen = candidate
            break

    words = chosen.split()
    if len(words) > 8:
        chosen = " ".join(words[:8])
    return chosen.strip(" -,:;").capitalize()


def postprocess_summary(summary: str, source_text: str) -> str:
    """Stabilize zero-shot titles for demo-friendly display."""
    cleaned_summary = clean_text(summary)
    lowered = cleaned_summary.lower()
    invalid_markers = ("warning:", "graphic content", "customer review", "review:")
    if (
        not cleaned_summary
        or any(marker in lowered for marker in invalid_markers)
        or len(cleaned_summary.split()) > 10
    ):
        return build_fallback_title(source_text)
    return cleaned_summary


@st.cache_data(show_spinner=False)
def summarize_batch(texts: tuple[str, ...]) -> list[str]:
    """Generate summaries for a small batch of reviews."""
    return [summarize_issue(text) for text in texts]


def render_review_cards(dataframe: pd.DataFrame) -> None:
    """Render a list of representative review cards."""
    if dataframe.empty:
        st.info("There are no representative negative reviews in the current scope yet.")
        return

    for _, row in dataframe.iterrows():
        excerpt = str(row.get("clean_review_text", "")).strip()
        if len(excerpt) > 260:
            excerpt = excerpt[:257] + "..."
        title = str(row.get("clean_review_title", "")).strip() or "Untitled review"
        st.markdown(
            f"""
            <div class="review-card">
                <div class="review-title">{title}</div>
                <div class="review-meta">
                    Rating {float(row.get("rating", 0)):.1f}
                    | Helpful votes {int(row.get("helpful_votes", 0))}
                    | Value label {int(row.get("review_value_label", 0))}
                </div>
                <div class="review-body">{excerpt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ai_titles(dataframe: pd.DataFrame, scope_key: str) -> None:
    """Generate and render AI complaint titles for representative reviews."""
    if dataframe.empty:
        st.info("There are no negative reviews available for AI title generation in the current scope.")
        return

    review_rows = dataframe.head(3).copy()
    button_key = f"ai_titles_{scope_key}"
    if st.button("Generate AI Complaint Titles", key=button_key):
        with st.spinner("Generating concise issue titles..."):
            texts = tuple(review_rows["clean_review_text"].fillna("").tolist())
            st.session_state[f"generated_titles_{scope_key}"] = summarize_batch(texts)

    titles = st.session_state.get(f"generated_titles_{scope_key}")
    if not titles:
        st.caption("Click the button to generate AI complaint titles for representative negative reviews in the current scope.")
        return

    for generated_title, (_, row) in zip(titles, review_rows.iterrows()):
        st.markdown(
            f"""
            <div class="section-card" style="margin-bottom:0.7rem;">
                <strong>AI title:</strong> {generated_title}<br/>
                <span style="color:#7b6c5e;">Reference title:</span> {row.get("clean_review_title", "N/A")}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_live_analyzer() -> None:
    """Render the single-review analysis experience."""
    st.subheader("Live Review Analyzer")
    st.caption("Paste one review to run the full pipeline: review value, sentiment, and title generation.")
    review_text = st.text_area(
        "Customer review",
        height=170,
        placeholder="Paste a customer review here for a live demo...",
    )

    if not st.button("Analyze Review", type="primary"):
        return

    cleaned_text = clean_text(review_text)
    if not cleaned_text:
        st.warning("Please enter a review before running the analyzer.")
        return

    with st.spinner("Running the local models..."):
        value_result = predict_review_value(cleaned_text)
        sentiment_result = predict_sentiment(cleaned_text)
        generated_title = summarize_issue(cleaned_text) if sentiment_result["label"] == "negative" else ""

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric(
        "Review Value",
        "High" if value_result["label"] == 1 else "Low",
        f"{value_result['probability']:.1%} probability",
    )
    metric_col2.metric(
        "Sentiment",
        sentiment_result["label"].capitalize(),
        f"negative {sentiment_result['negative_probability']:.1%}",
    )
    metric_col3.metric(
        "Summary",
        generated_title or "Not generated",
        "negative-only",
    )

    st.markdown(
        f"""
        <div class="section-card">
            <strong>Cleaned review</strong><br/>
            {cleaned_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_samples(samples: list[dict[str, str]]) -> None:
    """Render a few curated summary examples."""
    if not samples:
        st.info("No summary samples available yet.")
        return
    for sample in samples[:3]:
        st.markdown(
            f"""
            <div class="section-card" style="margin-bottom:0.7rem;">
                <strong>Generated title:</strong> {sample.get("generated_summary", "")}<br/>
                <span style="color:#7b6c5e;">Reference title:</span> {sample.get("target_text", "")}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat_bubble(role: str, content: str) -> None:
    """Render a GPT-style chat bubble."""
    role_class = "user" if role == "user" else "assistant"
    role_label = "You" if role == "user" else "Assistant"
    safe_content = escape(content).replace("\n", "<br/>")
    st.markdown(
        f"""
        <div class="chat-row {role_class}">
            <div class="chat-bubble {role_class}">
                <div class="chat-role">{role_label}</div>
                <div>{safe_content}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat(scope_dataframe: pd.DataFrame, *, scope_label: str, use_external_llm: bool) -> None:
    """Render the simple dialogue component."""
    st.subheader("Merchant Assistant")
    st.caption(
        "Ask about the current category or product. The assistant works offline by default "
        "and can optionally use an external API if configured."
    )

    chat_ui_version = "assistant_en_v4"
    if st.session_state.get("chat_ui_version") != chat_ui_version:
        st.session_state["chat_ui_version"] = chat_ui_version
        st.session_state["chat_scope_key"] = None
        st.session_state["chat_messages"] = []

    suggestion_columns = st.columns(3)
    for index, (column, suggestion) in enumerate(
        zip(suggestion_columns, get_chat_suggestions(scope_label))
    ):
        if column.button(
            suggestion,
            use_container_width=True,
            key=f"chat_suggestion_{chat_ui_version}_{scope_label}_{index}",
        ):
            st.session_state["pending_chat_question"] = suggestion

    scope_key = scope_label
    if st.session_state.get("chat_scope_key") != scope_key:
        st.session_state["chat_scope_key"] = scope_key
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    f"You are currently viewing '{scope_label}'. "
                    "Ask about review counts, complaint themes, products, or representative negative reviews."
                ),
            }
        ]

    history_container = st.container(height=460, border=True)
    with history_container:
        for message in st.session_state.get("chat_messages", []):
            render_chat_bubble(message["role"], message["content"])

    pending_question = st.session_state.pop("pending_chat_question", None)
    user_question = st.chat_input("Message the assistant about this scope...")
    question = pending_question or user_question
    if not question:
        return

    st.session_state["chat_messages"].append({"role": "user", "content": question})
    with st.spinner("Thinking..."):
        answer = answer_chat_question(
            question,
            scope_dataframe,
            scope_label=scope_label,
            use_external_llm=use_external_llm,
        )
    st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
    st.rerun()


def main() -> None:
    """Launch the dashboard application."""
    st.set_page_config(
        page_title="Review Insight Studio",
        page_icon="R",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    payload = load_payload()
    metrics = load_metrics()
    merged_reviews = payload["merged_reviews"]
    overview_snapshot = payload["overview_snapshot"]

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Review Intelligence Pipeline</div>
            <div class="hero-title">Review Insight Studio</div>
            <p class="hero-copy">
                Explore high-value reviews, surface negative issues, generate short complaint titles,
                and test a simple merchant-facing dialogue component in one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Filter Scope")
    split = st.sidebar.selectbox("Dataset split", ["all", "train", "validation", "test"])
    category = st.sidebar.selectbox("Category", payload["category_options"])

    product_choices = payload["products_by_category"].get(category, [("All", "All products")])
    product_labels = [label if product_id == "All" else f"{label} ({product_id})" for product_id, label in product_choices]
    selected_label = st.sidebar.selectbox("Product", product_labels)
    selected_index = product_labels.index(selected_label)
    selected_product_id, selected_product_label = product_choices[selected_index]

    openai_ready = bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL"))
    use_external_llm = st.sidebar.toggle(
        "Use external LLM in chat",
        value=False,
        disabled=not openai_ready,
        help=(
            "Enable this only if OPENAI_API_KEY and OPENAI_MODEL are configured. "
            "Otherwise the chat assistant stays fully local."
        ),
    )
    if not openai_ready:
        st.sidebar.caption("External API chat is optional and currently not configured.")

    scope_dataframe = filter_scope_dataframe(
        merged_reviews,
        category=category,
        product_id=selected_product_id,
        split=split,
    )
    scope_label = (
        selected_product_label
        if selected_product_id != "All"
        else (category if category != "All" else "Entire evaluation set")
    )
    scope_key = f"{split}_{category}_{selected_product_id}"
    scope_snapshot = build_scope_snapshot(scope_dataframe, scope_label=scope_label)

    tab_overview, tab_explorer, tab_live, tab_chat = st.tabs(
        ["Overview", "Explorer", "Live Analyzer", "Assistant"]
    )

    with tab_overview:
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric(
            "Best Test Result: Review Value Accuracy",
            f"{metrics['review_value']['test']['accuracy']:.1%}",
        )
        result_col2.metric(
            "Best Test Result: BERT Sentiment F1",
            f"{metrics['sentiment']['test']['test_f1']:.3f}",
        )
        result_col3.metric(
            "Summary Demo Mode",
            "Zero-shot T5",
            "sample-based showcase",
        )

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Processed Reviews", f"{overview_snapshot['total_reviews']:,}")
        metric_col2.metric("High-value Reviews", f"{overview_snapshot['high_value_reviews']:,}")
        metric_col3.metric("Negative Reviews", f"{overview_snapshot['negative_reviews']:,}")
        metric_col4.metric(
            "High-value Negative",
            f"{overview_snapshot['high_value_negative_reviews']:,}",
        )

        chart_col1, chart_col2 = st.columns([1.2, 1])
        with chart_col1:
            st.plotly_chart(
                build_category_overview_figure(merged_reviews),
                use_container_width=True,
                key="overview_category_chart",
            )
        with chart_col2:
            keyword_figure = build_keyword_figure(overview_snapshot["top_keywords"])
            if keyword_figure is not None:
                st.plotly_chart(
                    keyword_figure,
                    use_container_width=True,
                    key="overview_keyword_chart",
                )
            else:
                st.info("No complaint keywords available yet.")

        category_table = (
            merged_reviews.groupby("category")
            .agg(
                total_reviews=("review_id", "count"),
                high_value_reviews=("review_value_label", "sum"),
                negative_reviews=("sentiment_label", lambda series: int((series == "negative").sum())),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
            .sort_values(by="negative_reviews", ascending=False)
        )
        st.dataframe(category_table, use_container_width=True, hide_index=True)

        st.markdown("**Generated Complaint Title Samples**")
        render_summary_samples(metrics["summary_samples"].get("test", []))

    with tab_explorer:
        scope_metric1, scope_metric2, scope_metric3, scope_metric4 = st.columns(4)
        scope_metric1.metric("Current Scope", scope_label)
        scope_metric2.metric("Reviews", f"{scope_snapshot['total_reviews']:,}")
        scope_metric3.metric("Negative", f"{scope_snapshot['negative_reviews']:,}")
        scope_metric4.metric(
            "Verified Purchase Rate",
            f"{scope_snapshot['verified_purchase_rate']:.0%}",
        )

        figure_col, review_col = st.columns([1, 1.1])
        with figure_col:
            keyword_figure = build_keyword_figure(scope_snapshot["top_keywords"])
            if keyword_figure is not None:
                st.plotly_chart(
                    keyword_figure,
                    use_container_width=True,
                    key=f"explorer_keyword_chart_{scope_key}",
                )
            else:
                st.info("No complaint keywords available for the current scope.")

            st.markdown(
                f"""
                <div class="section-card">
                    <strong>Scope summary</strong><br/>
                    Average rating: {scope_snapshot['average_rating']:.2f}<br/>
                    High-value reviews: {scope_snapshot['high_value_reviews']}<br/>
                    Calibrated sentiment reviews: {scope_snapshot['calibrated_sentiment_reviews']}<br/>
                    High-value negative reviews: {scope_snapshot['high_value_negative_reviews']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with review_col:
            st.markdown("**Representative Negative Reviews**")
            negative_reviews = scope_dataframe[scope_dataframe["sentiment_label"] == "negative"]
            representative_reviews = (
                scope_snapshot["representative_reviews"]
                if not scope_snapshot["representative_reviews"].empty
                else negative_reviews.head(3)
            )
            render_review_cards(representative_reviews)

        st.markdown("**AI Complaint Titles**")
        render_ai_titles(negative_reviews, scope_key)

    with tab_live:
        render_live_analyzer()

    with tab_chat:
        render_chat(
            scope_dataframe,
            scope_label=scope_label,
            use_external_llm=use_external_llm,
        )


if __name__ == "__main__":
    main()
