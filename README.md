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
   - Model: zero-shot `google/flan-t5-small`.
   - This is used as a demo component, not as a strongly evaluated fine-tuned summarizer.

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

models/
  review_value_classifier.pkl      # TF-IDF + Logistic Regression bundle
  review_value_metrics.json        # Review-value metrics
  bert_sentiment/                  # Fine-tuned BERT model
  bert_sentiment_metrics.json      # BERT metrics
  t5_summary_metrics.json          # Zero-shot T5 metrics
  t5_summary_samples.json          # Example generated titles
  t5_summary/                      # Legacy T5 experiment output, not used by the dashboard by default

src/
  preprocessing/                   # Text cleaning and generic data preparation
  helpfulness/                     # Review-value dataset, training, prediction
  sentiment/                       # Sentiment label calibration and BERT training
  summarization/                   # T5 generation and zero-shot evaluation
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

Current overview counts:

| Metric | Count |
| --- | ---: |
| Total processed reviews | 4,000 |
| High-value reviews | 600 |
| Calibrated negative reviews | 305 |
| High-value negative reviews | 47 |

## Current Model Results

| Task | Main test result | Notes |
| --- | ---: | --- |
| Review value classification | Accuracy 0.840 | Positive-class F1 is about 0.418, so high-value review detection is still the weak point. |
| BERT sentiment classification | F1 0.990 | Strong on the calibrated labels, but the labels are rule-derived. |
| T5 complaint title generation | Avg unigram F1 0.086 | Zero-shot demo baseline; exact match is 0. |

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

The assistant tab can optionally call an external OpenAI-compatible model if both environment variables are set:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="your-model-name"
.venv/bin/streamlit run app/streamlit_app.py
```

If those variables are not set, the assistant uses local rule-based answers from the dashboard data.

Do not start the app with `python app/streamlit_app.py`; Streamlit apps must be launched with `streamlit run`.

The dashboard expects the processed CSV files and saved model outputs in `data/processed/` and `models/`. The complaint-title component uses the cached `google/flan-t5-small` checkpoint in zero-shot mode; it does not fine-tune or load the legacy `models/t5_summary/` experiment directory during the dashboard demo.

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

Train the review-value classifier:

```bash
.venv/bin/python -m src.helpfulness.train_helpfulness
```

Fine-tune the BERT sentiment classifier:

```bash
.venv/bin/python -m src.sentiment.train_bert_sentiment
```

Evaluate the zero-shot T5 complaint-title baseline:

```bash
.venv/bin/python -m src.summarization.train_t5 --allow-download
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
  --model-name google/flan-t5-small \
  --allow-download \
  --text "The caps do not stay on the pencils and the tips keep getting ruined."
```

If the model is already cached locally, `--allow-download` can be omitted.

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
- The T5 summarization component is zero-shot. It is useful for a demo, but the saved metrics show that it is not a strong summarization model.
- The local `models/` directory is large because it includes transformer model weights and checkpoints. The Git version intentionally ignores large model directories such as `models/bert_sentiment/` and `models/t5_summary/`; keep them locally, publish them with Git LFS, or recreate them by rerunning the training/evaluation commands.
