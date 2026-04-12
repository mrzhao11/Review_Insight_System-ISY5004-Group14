# Review Insight System

Review Insight System is a coursework-style machine learning project for analyzing Amazon customer reviews. It builds a small review intelligence pipeline that helps merchants identify high-value reviews, detect negative sentiment, generate short complaint titles, and explore the results in a Streamlit dashboard.

## What This Project Does

The project uses Amazon Reviews 2023 data and focuses on three modeling tasks:

1. Review value classification
   - Goal: predict whether a review is likely to be high-value.
   - Label rule: `review_value_label = 1` when `helpful_votes >= 2`, otherwise `0`.
   - Model: TF-IDF features with Logistic Regression.

2. Sentiment classification
   - Goal: classify review text as positive or negative.
   - Labels are calibrated from star ratings plus VADER sentiment scores, not manually annotated.
   - Positive rule: `rating >= 4` and VADER compound score above `0.0`.
   - Negative rule: `rating <= 2` and VADER compound score below `0.05`.
   - Model: fine-tuned `bert-base-uncased`.

3. Complaint title generation
   - Goal: generate a short title for negative reviews.
   - Preprocessing step: Volcengine Ark generates normalized pseudo complaint titles for calibrated negative reviews.
   - Student model: fine-tuned `google/flan-t5-small`, saved locally in `models/t5_pseudo_summary/`.
   - Runtime fallback: local student first, Ark direct generation second, zero-shot T5 or a local heuristic last.

The Streamlit app combines these pieces into an interactive dashboard with category filters, representative negative reviews, live single-review analysis, and a simple merchant assistant.

## Repository Layout

```text
app/
  streamlit_app.py                 # Streamlit dashboard entry point

data/processed/
  review_value_train.csv           # Review-value train split
  review_value_validation.csv      # Review-value validation split
  review_value_test.csv            # Review-value test split
  review_value_manifest.json       # Dataset sampling and label manifest
  sentiment_train.csv              # Calibrated sentiment train split
  sentiment_validation.csv         # Calibrated sentiment validation split
  sentiment_test.csv               # Calibrated sentiment test split
  sentiment_manifest.json          # Sentiment calibration manifest
  pseudo_summary_train.csv         # Ark pseudo-title train split
  pseudo_summary_validation.csv    # Ark pseudo-title validation split
  pseudo_summary_test.csv          # Ark pseudo-title test split
  pseudo_summary_manifest.json     # Ark pseudo-title generation manifest

models/
  review_value_classifier.pkl      # TF-IDF + Logistic Regression bundle
  review_value_metrics.json        # Review-value metrics
  bert_sentiment/                  # Fine-tuned BERT model
  bert_sentiment_metrics.json      # BERT metrics
  t5_pseudo_summary/               # T5 student fine-tuned on Ark pseudo titles
  t5_pseudo_summary_metrics.json   # Pseudo-label T5 student metrics
  t5_pseudo_summary_samples.json   # Example pseudo-label T5 generations
  t5_summary_metrics.json          # Zero-shot T5 metrics
  t5_summary_samples.json          # Example zero-shot generated titles
  t5_summary/                      # Legacy T5 experiment output

src/
  preprocessing/                   # Text cleaning and generic data preparation
  helpfulness/                     # Review-value dataset, training, prediction
  sentiment/                       # Sentiment label calibration and BERT training
  summarization/                   # Ark pseudo-title internals, T5 training, and generation
  visualization/                   # Dashboard data, charts, and chat utilities
  utils/                           # Amazon Reviews JSONL loading helpers
```

## Current Processed Dataset

The checked-in processed data contains 4,000 reviews across five categories:

- `All_Beauty`
- `Amazon_Fashion`
- `Appliances`
- `Handmade_Products`
- `Health_and_Personal_Care`

Current split sizes:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Review value | 3,200 | 400 | 400 |
| Calibrated sentiment | 964 | 319 | 329 |
| Ark pseudo complaint titles | 214 | 30 | 61 |

Current overview counts:

| Metric | Count |
| --- | ---: |
| Total processed reviews | 4,000 |
| High-value reviews | 600 |
| Calibrated negative reviews | 305 |
| High-value negative reviews | 47 |
| Ark pseudo complaint-title pairs | 305 |

## Current Model Results

| Task | Main test result | Notes |
| --- | ---: | --- |
| Review value classification | Accuracy 0.840 | Positive-class F1 is about 0.418, so high-value review detection is still the weak point. |
| BERT sentiment classification | F1 0.990 | Strong on the calibrated labels, but the labels are rule-derived. |
| T5 complaint title generation | ROUGE-L F1 0.334 | Fine-tuned on Ark-generated pseudo titles; zero-shot ROUGE-L F1 on the same expanded test split is 0.146. |

Current complaint-title metrics use an expanded 61-row test split. Exact-match is intentionally not reported because short complaint titles can have multiple valid phrasings.

## Setup

The project was last run with Python 3.9.6. A local virtual environment already exists in `.venv`.

To install dependencies in a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run The Dashboard

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

The dashboard runs fully local by default.

The assistant tab can optionally call Volcengine Ark through its OpenAI-compatible API. Configure the Ark API key before launching Streamlit:

```bash
export ARK_API_KEY="your-ark-api-key"
.venv/bin/streamlit run app/streamlit_app.py
```

You can also copy `.env.example` to `.env` and fill in your local key. `.env` is ignored by Git:

```bash
cp .env.example .env
```

`ARK_MODEL` can be any enabled Ark text-generation model available to your account. This project currently uses `doubao-seed-2-0-lite-260215` when `ARK_MODEL` is not set:

```bash
export ARK_MODEL="doubao-seed-2-0-lite-260215"
export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
```

If `ARK_API_KEY` is not set, the assistant uses local rule-based answers from the dashboard data. The API key is never stored in the repository; configure it in your terminal or deployment environment.

Do not start the app with `python app/streamlit_app.py`; Streamlit apps must be launched with `streamlit run`.

The dashboard expects the processed CSV files and saved model outputs in `data/processed/` and `models/`. The complaint-title component prefers the local pseudo-label student in `models/t5_pseudo_summary/`. If that directory is missing, it falls back to Ark direct title generation when `ARK_API_KEY` is configured. If Ark is unavailable, it falls back to the cached zero-shot `google/flan-t5-small` checkpoint and then a local rule-based title.

## Rebuild The Pipeline

Run commands from the project root.

Build the review-value dataset:

```bash
.venv/bin/python -m src.helpfulness.prepare_helpfulness_dataset \
  --categories All_Beauty Amazon_Fashion Appliances Handmade_Products Health_and_Personal_Care \
  --samples-per-category 800 \
  --min-helpful-per-category 120 \
  --helpful-vote-threshold 2 \
  --limit 20000
```

Calibrate sentiment labels:

```bash
.venv/bin/python -m src.sentiment.label_calibration
```

Generate complaint-title pseudo labels as preprocessing:

```bash
.venv/bin/python -m src.preprocessing.generate_complaint_titles
```

This command requires `ARK_API_KEY`. It writes `data/processed/pseudo_summary_train.csv`, `data/processed/pseudo_summary_validation.csv`, `data/processed/pseudo_summary_test.csv`, and `data/processed/pseudo_summary_manifest.json`. Use `--limit 2` for a cheap smoke test before running the full 305 negative reviews.

Re-split the generated complaint titles to keep a larger test set:

```bash
.venv/bin/python -m src.preprocessing.resplit_complaint_titles
```

The default re-split is 70% train, 10% validation, and 20% test, producing 214/30/61 rows from the 305 generated pseudo-title pairs.

Train the review-value classifier:

```bash
.venv/bin/python -m src.helpfulness.train_helpfulness
```

Fine-tune the BERT sentiment classifier:

```bash
.venv/bin/python -m src.sentiment.train_bert_sentiment
```

Fine-tune the T5 complaint-title student on the preprocessed Ark pseudo titles:

```bash
.venv/bin/python -m src.summarization.fine_tune_t5_pseudo --allow-download
```

If `google/flan-t5-small` is already cached locally, `--allow-download` can be omitted. The student model is written to `models/t5_pseudo_summary/`, with metrics and samples saved as top-level JSON files in `models/`.

Evaluate the zero-shot T5 complaint-title baseline:

```bash
.venv/bin/python -m src.summarization.train_t5 \
  --train-file data/processed/pseudo_summary_train.csv \
  --validation-file data/processed/pseudo_summary_validation.csv \
  --test-file data/processed/pseudo_summary_test.csv \
  --target-column llm_complaint_title \
  --allow-download
```

If `google/flan-t5-small` is already cached locally, `--allow-download` can be omitted.

## Single-Task Utilities

Predict review value for one text:

```bash
.venv/bin/python -m src.helpfulness.predict_helpfulness \
  --text "This product broke after one day and the instructions were useless."
```

Generate a complaint title for one text:

```bash
.venv/bin/python -m src.summarization.generate_summary \
  --model-dir models/t5_pseudo_summary \
  --text "The caps do not stay on the pencils and the tips keep getting ruined."
```

To compare with the zero-shot baseline, use `--model-name google/flan-t5-small --allow-download` instead of `--model-dir`.

## Demo Test Inputs

Use these examples in the dashboard's Live Analyzer:

```text
This blender broke after one week and customer service never replied to me.
```

```text
The product looked promising at first, but after three days the battery stopped charging and the screen started flickering. I contacted support twice and got no response.
```

```text
I love this product. It works exactly as described, feels high quality, and arrived on time.
```

Use these examples in the Assistant tab:

```text
How many high-value negative reviews are in Entire evaluation set?
```

```text
What are the main complaint themes in All_Beauty?
```

```text
Show me two representative negative review examples.
```

## Important Caveats

- `review_value_label` is a proxy for review helpfulness based on helpful vote count. It is not a human quality label.
- Sentiment labels are calibrated from ratings and VADER scores. The BERT score reflects performance on those rule-derived labels.
- The T5 complaint-title student is trained on LLM-generated pseudo labels, not human-written gold complaint labels. Treat its metrics as a demo-oriented signal.
- The local `models/` directory is large because it includes transformer model weights and checkpoints. The Git version intentionally ignores large model directories such as `models/bert_sentiment/`, `models/t5_summary/`, and `models/t5_pseudo_summary/`; keep them locally, publish them with Git LFS, or recreate them by rerunning the training/evaluation commands.
