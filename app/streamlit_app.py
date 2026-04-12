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

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime fallback
    OpenAI = None

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
SUMMARY_STUDENT_DIR = MODELS_DIR / "t5_pseudo_summary"
SUMMARY_MODEL_NAME = "google/flan-t5-small"
DEMO_SPLIT = "test"
UPLOAD_REQUIRED_COLUMNS = ("review_text",)
UPLOAD_OPTIONAL_COLUMNS = (
    "review_title",
    "rating",
    "product_id",
    "product_title",
    "category",
    "helpful_votes",
    "verified_purchase",
)
UPLOAD_ANALYSIS_LIMIT = 50
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_DEMO_MODEL = "doubao-seed-2-0-lite-260215"
CATEGORY_DISPLAY_NAMES = {
    "All": "All Categories",
    "All_Beauty": "Beauty Products",
    "Amazon_Fashion": "Fashion & Apparel",
    "Appliances": "Home Appliances",
    "Handmade_Products": "Handmade Products",
    "Health_and_Personal_Care": "Health & Personal Care",
}


def load_local_env(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file without overriding exports."""
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


load_local_env(PROJECT_ROOT / ".env")


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
    zero_shot_metrics = json.loads((MODELS_DIR / "t5_summary_metrics.json").read_text())
    zero_shot_samples = json.loads((MODELS_DIR / "t5_summary_samples.json").read_text())
    pseudo_metrics_path = MODELS_DIR / "t5_pseudo_summary_metrics.json"
    pseudo_samples_path = MODELS_DIR / "t5_pseudo_summary_samples.json"

    if pseudo_metrics_path.exists() and pseudo_samples_path.exists():
        summary_metrics = json.loads(pseudo_metrics_path.read_text())
        summary_samples = json.loads(pseudo_samples_path.read_text())
        summary_display_name = "Pseudo-label T5 student"
        summary_display_delta = (
            f"test unigram F1 {summary_metrics['test']['avg_unigram_f1']:.3f}"
        )
    else:
        summary_metrics = zero_shot_metrics
        summary_samples = zero_shot_samples
        summary_display_name = "Zero-shot T5"
        summary_display_delta = (
            f"test unigram F1 {zero_shot_metrics['test']['avg_unigram_f1']:.3f}"
        )

    return {
        "review_value": json.loads((MODELS_DIR / "review_value_metrics.json").read_text()),
        "sentiment": json.loads((MODELS_DIR / "bert_sentiment_metrics.json").read_text()),
        "summary": summary_metrics,
        "summary_samples": summary_samples,
        "summary_display_name": summary_display_name,
        "summary_display_delta": summary_display_delta,
        "zero_shot_summary": zero_shot_metrics,
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
def load_t5_summary_bundle(
    model_source: str,
) -> tuple[T5Tokenizer, T5ForConditionalGeneration, torch.device]:
    """Load a local or cached T5 complaint-title model."""
    resolved_source: Path | str = (
        Path(model_source) if Path(model_source).exists() else model_source
    )
    tokenizer = T5Tokenizer.from_pretrained(resolved_source, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        resolved_source,
        local_files_only=True,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def has_summary_student() -> bool:
    """Return whether the pseudo-label T5 student is available locally."""
    return (SUMMARY_STUDENT_DIR / "config.json").exists()


def generate_title_with_t5(text: str, *, model_source: str) -> str:
    """Generate a complaint title with a specific T5 model source."""
    tokenizer, model, device = load_t5_summary_bundle(model_source)
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
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def generate_title_with_ark(text: str) -> str | None:
    """Generate a complaint title with Ark when the local student is unavailable."""
    api_key = os.getenv("ARK_API_KEY")
    model_name = os.getenv("ARK_MODEL", ARK_DEMO_MODEL)
    base_url = os.getenv("ARK_BASE_URL", ARK_BASE_URL)
    if not api_key or not model_name or OpenAI is None:
        return None

    prompt = (
        "Write one short English complaint title for this negative e-commerce review.\n"
        "Rules:\n"
        "- Use 3 to 8 words.\n"
        "- Focus on the concrete product or service problem.\n"
        "- Use only facts stated in the review.\n"
        "- Do not mention star ratings.\n"
        "- Return only the title text.\n\n"
        f"Review: {text.strip()}"
    )

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=30.0,
        )
        response = client.responses.create(
            model=model_name,
            input=prompt,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return (response.output_text or "").strip() or None
    except Exception:
        return None


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
    """Generate a complaint title with student, Ark, zero-shot T5, then rules."""
    if has_summary_student():
        try:
            generated = generate_title_with_t5(
                text,
                model_source=str(SUMMARY_STUDENT_DIR),
            )
            return postprocess_summary(generated, text)
        except Exception:
            pass

    ark_title = generate_title_with_ark(text)
    if ark_title:
        return postprocess_summary(ark_title, text)

    try:
        generated = generate_title_with_t5(text, model_source=SUMMARY_MODEL_NAME)
        return postprocess_summary(generated, text)
    except Exception:
        return build_fallback_title(text)


def build_fallback_title(text: str) -> str:
    """Create a compact fallback issue title when generated output is noisy."""
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
    """Stabilize generated titles for demo-friendly display."""
    cleaned_summary = clean_text(summary)
    lowered = cleaned_summary.lower()
    source_lowered = source_text.lower()
    invalid_markers = ("warning:", "graphic content", "customer review", "review:")
    source_negation_patterns = (
        r"\bnot\b",
        r"\bnever\b",
        r"\bno\b",
        r"n't\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bwon't\b",
    )
    problem_signal_patterns = (
        *source_negation_patterns,
        r"\bfail\w*\b",
        r"\bbroke\b",
        r"\bbroken\b",
        r"\bstopped\b",
        r"\bwaste\b",
        r"\bpoor\b",
        r"\bcheap\b",
        r"\bwrong\b",
        r"\bissue\b",
        r"\bproblem\b",
        r"\btoo\b",
        r"\bloose\b",
        r"\bscratch\w*\b",
        r"\bcrack\w*\b",
        r"\bfall\w*\b",
        r"\bmissing\b",
        r"\binoperable\b",
        r"\bhard\b",
        r"\bdifficult\b",
    )
    source_has_negation = any(
        re.search(pattern, source_lowered) for pattern in source_negation_patterns
    )
    summary_has_problem_signal = any(
        re.search(pattern, lowered) for pattern in problem_signal_patterns
    )
    if (
        not cleaned_summary
        or any(marker in lowered for marker in invalid_markers)
        or len(cleaned_summary.split()) > 10
        or (source_has_negation and not summary_has_problem_signal)
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


def build_upload_template() -> pd.DataFrame:
    """Create a strict CSV template for merchant uploads."""
    return pd.DataFrame(
        [
            {
                "review_text": "The bottle leaked after two days and the replacement cap did not fit.",
                "review_title": "Leaking bottle",
                "rating": 2,
                "product_id": "SKU-001",
                "product_title": "Insulated Water Bottle",
                "category": "Drinkware",
                "helpful_votes": 0,
                "verified_purchase": True,
            },
            {
                "review_text": "Arrived on time, works well, and the finish feels sturdy.",
                "review_title": "Solid purchase",
                "rating": 5,
                "product_id": "SKU-002",
                "product_title": "Desk Lamp",
                "category": "Home Office",
                "helpful_votes": 1,
                "verified_purchase": True,
            },
        ],
        columns=[*UPLOAD_REQUIRED_COLUMNS, *UPLOAD_OPTIONAL_COLUMNS],
    )


def build_product_choices(dataframe: pd.DataFrame, category: str) -> list[tuple[str, str]]:
    """Build product filter choices from the current demo scope."""
    scoped = dataframe if category == "All" else dataframe[dataframe["category"] == category]
    if scoped.empty:
        return [("All", "All products")]

    product_frame = (
        scoped[["product_id", "product_title"]]
        .drop_duplicates()
        .fillna("")
        .sort_values(by=["product_title", "product_id"])
    )
    choices = [("All", "All products")]
    for product_id, product_title in product_frame.itertuples(index=False, name=None):
        product_id = str(product_id).strip()
        product_title = str(product_title).strip() or "Untitled product"
        if product_id:
            choices.append((product_id, product_title))
    return choices


def clean_upload_value(value: Any) -> str:
    """Clean a possibly-empty uploaded CSV cell."""
    if pd.isna(value):
        return ""
    return clean_text(str(value))


def format_category_label(category: Any) -> str:
    """Map dataset category codes to merchant-friendly labels."""
    category_text = str(category)
    return CATEGORY_DISPLAY_NAMES.get(category_text, category_text.replace("_", " "))


def with_display_categories(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with category codes replaced by presentation labels."""
    display_frame = dataframe.copy()
    if "category" in display_frame.columns:
        display_frame["category"] = display_frame["category"].map(format_category_label)
    return display_frame


def prepare_uploaded_reviews(uploaded_dataframe: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Validate and normalize a merchant-uploaded review CSV."""
    dataframe = uploaded_dataframe.copy()
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    missing_columns = [
        column for column in UPLOAD_REQUIRED_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        return None, missing_columns

    dataframe["clean_review_text"] = dataframe["review_text"].map(clean_upload_value)
    title_source = (
        dataframe["review_title"]
        if "review_title" in dataframe.columns
        else pd.Series([""] * len(dataframe), index=dataframe.index)
    )
    dataframe["clean_review_title"] = title_source.map(clean_upload_value)
    dataframe = dataframe[dataframe["clean_review_text"] != ""].reset_index(drop=True)
    return dataframe, []


def analyze_uploaded_reviews(dataframe: pd.DataFrame, row_limit: int) -> pd.DataFrame:
    """Run local models over uploaded merchant reviews."""
    analysis_rows: list[dict[str, Any]] = []
    for _, row in dataframe.head(row_limit).iterrows():
        text = str(row["clean_review_text"])
        value_result = predict_review_value(text)
        sentiment_result = predict_sentiment(text)
        generated_title = (
            summarize_issue(text) if sentiment_result["label"] == "negative" else ""
        )
        analysis_rows.append(
            {
                "review_text": text,
                "review_value": "High" if value_result["label"] == 1 else "Low",
                "review_value_probability": round(value_result["probability"], 4),
                "sentiment": str(sentiment_result["label"]).capitalize(),
                "negative_probability": round(
                    float(sentiment_result["negative_probability"]),
                    4,
                ),
                "generated_complaint_title": generated_title,
                "product_id": row.get("product_id", ""),
                "product_title": row.get("product_title", ""),
                "rating": row.get("rating", ""),
            }
        )
    return pd.DataFrame(analysis_rows)


def render_merchant_upload() -> None:
    """Render the merchant CSV upload interface."""
    st.subheader("Merchant Review Upload")
    st.caption(
        "Upload a merchant review CSV to run the local review-value, sentiment, "
        "and complaint-title pipeline on your own data."
    )

    st.info(
        "CSV format requirement: the file must be UTF-8 CSV and must include a "
        "`review_text` column. Optional columns are `review_title`, `rating`, "
        "`product_id`, `product_title`, `category`, `helpful_votes`, and "
        "`verified_purchase`. Column names are case-sensitive."
    )
    st.download_button(
        "Download CSV Template",
        data=build_upload_template().to_csv(index=False).encode("utf-8"),
        file_name="merchant_review_upload_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload merchant review CSV",
        type=["csv"],
        help="Required column: review_text. Keep the header row exactly as shown in the template.",
    )
    if uploaded_file is None:
        return

    try:
        uploaded_dataframe = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this CSV file: {exc}")
        return

    prepared_dataframe, missing_columns = prepare_uploaded_reviews(uploaded_dataframe)
    if missing_columns:
        st.error(
            "The uploaded file is missing required column(s): "
            + ", ".join(missing_columns)
        )
        return
    if prepared_dataframe is None or prepared_dataframe.empty:
        st.warning("No usable review text was found after cleaning the uploaded file.")
        return

    st.success(f"Loaded {len(prepared_dataframe):,} usable reviews.")
    preview_columns = [
        column
        for column in [
            "review_text",
            "review_title",
            "rating",
            "product_id",
            "product_title",
            "category",
        ]
        if column in prepared_dataframe.columns
    ]
    st.dataframe(
        prepared_dataframe[preview_columns].head(10),
        width="stretch",
        hide_index=True,
    )

    row_limit = st.slider(
        "Rows to analyze",
        min_value=1,
        max_value=min(len(prepared_dataframe), UPLOAD_ANALYSIS_LIMIT),
        value=min(len(prepared_dataframe), 10),
        help="Batch analysis is capped for demo speed because BERT and T5 run locally.",
    )
    if not st.button("Analyze Uploaded Reviews", type="primary"):
        return

    with st.spinner("Running local models on the uploaded reviews..."):
        result_dataframe = analyze_uploaded_reviews(prepared_dataframe, row_limit)

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Analyzed Rows", f"{len(result_dataframe):,}")
    result_col2.metric(
        "Predicted Negative",
        f"{int((result_dataframe['sentiment'] == 'Negative').sum()):,}",
    )
    result_col3.metric(
        "Predicted High Value",
        f"{int((result_dataframe['review_value'] == 'High').sum()):,}",
    )
    st.dataframe(result_dataframe, width="stretch", hide_index=True)
    st.download_button(
        "Download Analysis Results",
        data=result_dataframe.to_csv(index=False).encode("utf-8"),
        file_name="merchant_review_analysis_results.csv",
        mime="text/csv",
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


def render_chat(scope_dataframe: pd.DataFrame, *, scope_label: str, use_ark_llm: bool) -> None:
    """Render the simple dialogue component."""
    st.subheader("Merchant Copilot")
    st.caption(
        "Ask about the current category or product. Ark LLM can answer with richer "
        "merchant recommendations when configured; local rules remain available as fallback."
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
            width="stretch",
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
                    "Ask about review counts, complaint themes, risky products, or recommended actions."
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
            use_ark_llm=use_ark_llm,
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
    merged_reviews = filter_scope_dataframe(payload["merged_reviews"], split=DEMO_SPLIT)
    overview_snapshot = build_scope_snapshot(
        merged_reviews,
        scope_label="Held-out test set",
    )

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Merchant Review Intelligence</div>
            <div class="hero-title">Review Insight Studio</div>
            <p class="hero-copy">
                Monitor customer pain points, inspect high-value negative reviews,
                generate complaint titles, and ask for merchant-ready recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Filter Scope")
    st.sidebar.caption("Demo scope uses the held-out test set.")
    category_options = ["All"] + sorted(merged_reviews["category"].dropna().unique().tolist())
    category = st.sidebar.selectbox(
        "Category",
        category_options,
        format_func=format_category_label,
    )

    product_choices = build_product_choices(merged_reviews, category)
    product_labels = [label if product_id == "All" else f"{label} ({product_id})" for product_id, label in product_choices]
    selected_label = st.sidebar.selectbox("Product", product_labels)
    selected_index = product_labels.index(selected_label)
    selected_product_id, selected_product_label = product_choices[selected_index]

    ark_ready = bool(os.getenv("ARK_API_KEY"))
    ark_model = os.getenv("ARK_MODEL", ARK_DEMO_MODEL)
    use_ark_llm = st.sidebar.toggle(
        "Use Ark LLM in chat",
        value=False,
        disabled=not ark_ready,
        help=(
            "Enable this after configuring ARK_API_KEY. ARK_MODEL can be any "
            "enabled Ark text model; this project uses the demo model when it is not set."
        ),
    )
    if ark_ready:
        st.sidebar.caption(f"Current Ark chat model: {ark_model}")
    else:
        st.sidebar.caption("ARK_API_KEY is not configured. Chat is using local fallback.")

    scope_dataframe = filter_scope_dataframe(
        merged_reviews,
        category=category,
        product_id=selected_product_id,
        split=DEMO_SPLIT,
    )
    scope_label = (
        selected_product_label
        if selected_product_id != "All"
        else (format_category_label(category) if category != "All" else "Test Set")
    )
    scope_key = f"{DEMO_SPLIT}_{category}_{selected_product_id}"
    scope_snapshot = build_scope_snapshot(scope_dataframe, scope_label=scope_label)

    tab_overview, tab_explorer, tab_live, tab_upload, tab_chat = st.tabs(
        [
            "Business Overview",
            "Issue Explorer",
            "Single Review Check",
            "Merchant Upload",
            "Merchant Copilot",
        ]
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
            "Complaint Title Mode",
            metrics["summary_display_name"],
            metrics["summary_display_delta"],
        )

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Test Reviews", f"{overview_snapshot['total_reviews']:,}")
        metric_col2.metric("High-value Reviews", f"{overview_snapshot['high_value_reviews']:,}")
        metric_col3.metric("Negative Reviews", f"{overview_snapshot['negative_reviews']:,}")
        metric_col4.metric(
            "High-value Negative",
            f"{overview_snapshot['high_value_negative_reviews']:,}",
        )

        chart_col1, chart_col2 = st.columns([1.2, 1])
        with chart_col1:
            st.plotly_chart(
                build_category_overview_figure(with_display_categories(merged_reviews)),
                width="stretch",
                key="overview_category_chart",
            )
        with chart_col2:
            keyword_figure = build_keyword_figure(overview_snapshot["top_keywords"])
            if keyword_figure is not None:
                st.plotly_chart(
                    keyword_figure,
                    width="stretch",
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
        category_table["category"] = category_table["category"].map(format_category_label)
        category_table = category_table.rename(
            columns={
                "category": "Category",
                "total_reviews": "Reviews",
                "high_value_reviews": "High-value Reviews",
                "negative_reviews": "Negative Reviews",
                "avg_rating": "Average Rating",
            }
        )
        st.dataframe(category_table, width="stretch", hide_index=True)

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
                    width="stretch",
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

    with tab_upload:
        render_merchant_upload()

    with tab_chat:
        render_chat(
            scope_dataframe,
            scope_label=scope_label,
            use_ark_llm=use_ark_llm,
        )


if __name__ == "__main__":
    main()
