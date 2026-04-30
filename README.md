# Review Insight System

Review Insight System is a coursework ML project for Amazon review analytics.  
It delivers an end-to-end pipeline from sampled data construction to model training and a local Streamlit dashboard for business-facing review analysis.

## Business Problem

E-commerce merchants receive a large volume of reviews, but raw review text is difficult to use directly for decision-making. A merchant usually does not want to read thousands of reviews one by one just to answer a few practical questions:

1. Which reviews are likely to contain useful business signals?
2. Which reviews indicate negative customer experience?
3. What is the main complaint in a long negative review?

This project turns that business problem into a three-stage NLP pipeline and a local dashboard. The goal is not only to train individual models, but to build a usable review intelligence workflow for business exploration.

## Project At A Glance

This project targets three tasks:

1. Review value classification (`high-value` vs `low-value`)
2. Sentiment classification (`positive` vs `negative`)
3. Complaint title generation for negative reviews

The dashboard supports category/product filtering, representative complaint-candidate inspection, single-review inference, merchant CSV upload, and assistant-style Q&A.

## End-To-End Workflow

At a high level, the system works like this:

1. Sample and clean Amazon review data from five product categories.
2. Build a `review_value` dataset using helpful votes as a proxy business-value label.
3. Build a calibrated sentiment dataset using star ratings plus standard VADER compound thresholds.
4. Generate pseudo complaint titles only for negative reviews using an LLM teacher model.
5. Fine-tune local student models for sentiment classification and complaint-title generation.
6. Serve the trained components inside a Streamlit dashboard for interactive analysis.

In other words, this repository is both:

1. An experiment pipeline for dataset construction, training, and evaluation.
2. A lightweight business-facing application for exploring model outputs.

## Current Dataset Snapshot

Current processed scope is sampled from 5 categories:

- `All_Beauty`
- `Amazon_Fashion`
- `Appliances`
- `Handmade_Products`
- `Health_and_Personal_Care`

Current sizes:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Review value | 8,000 | 1,000 | 1,000 |
| Calibrated sentiment | 2,548 | 818 | 798 |
| Pseudo complaint titles | 560 | 80 | 160 |

Overview counts:

| Metric | Count |
| --- | ---: |
| Total processed reviews | 10,000 |
| High-value reviews | 1,250 |
| Calibrated negative reviews | 800 |
| Dashboard complaint candidates | 2,133 |
| High-value complaint candidates | 306 |
| Pseudo title pairs | 800 |

## Current Model Results

| Task | Test Result | Notes |
| --- | --- | --- |
| Review value (TF-IDF + LR) | Accuracy **0.795**; Positive F1 **0.255** | Positive-class recall remains the main weakness. |
| Sentiment (BERT) | Accuracy **0.966**; F1 **0.981**; Macro-F1 **0.910** | Strong performance on calibrated labels. |
| Title generation (Flan-T5-small, tuned) | ROUGE-1/2/L: **0.322 / 0.149 / 0.306**; BERTScore F1 **0.816** | Trained on pseudo titles. |
| Title generation (Flan-T5-base, tuned) | ROUGE-1/2/L: **0.399 / 0.178 / 0.373**; BERTScore F1 **0.830** | Best current title model. |

## What The Dashboard Does

The Streamlit app is designed as the final system surface rather than a separate demo page. It brings the processed data and trained models into one workflow:

1. `Business Overview`: shows key metrics, scope-level counts, category distributions, and summary business indicators.
2. `Issue Explorer`: lets the user inspect representative complaint candidates and generated complaint titles under a selected category or product.
3. `Single Review Check`: runs the full pipeline on one pasted review, including value classification, sentiment prediction, and complaint-title generation.
4. `Merchant Upload`: allows batch analysis of a merchant CSV file with local pipeline inference.
5. `Merchant Copilot`: provides assistant-style responses grounded in the selected review scope, with Ark API support when available.

This matters for assessment because the repository is not just a notebook-style experiment. It contains an integrated language processing system with both offline training and online usage paths.
## Reproducibility And Scope Policy

Important alignment policy:

- Complaint-title expansion must stay within the currently sampled `review_value_*.csv` universe.
- Do not add reviews from unseen products/categories during pseudo-title regeneration.

This keeps dataset scope consistent across preprocessing, training, and dashboard interpretation.

## Repository Structure

The repository is organized by pipeline stage so the reader can quickly map each folder to its role in the project.

```text
app/
  streamlit_app.py

data/processed/
  review_value_*.csv
  sentiment_*.csv
  pseudo_summary_*.csv
  *_manifest.json

models/
  review_value_classifier.pkl
  review_value_metrics.json
  bert_sentiment_metrics.json
  t5_pseudo_summary_small_metrics.json
  t5_pseudo_summary_base_metrics.json
  t5_pseudo_summary_small_samples.json
  t5_pseudo_summary_base_samples.json
  zero_vs_base_3sample_comparison.json

src/
  helpfulness/
  sentiment/
  summarization/
  preprocessing/
  visualization/
  utils/
```

### `app/`

- [app/streamlit_app.py](app/streamlit_app.py)
- Main entry point for the Streamlit dashboard.
- Loads trained models, merges processed datasets, handles user interaction, and exposes the business-facing analysis workflow.

### `data/processed/`

- Stores the prepared CSV splits used by downstream models and dashboard views.
- `review_value_*.csv`: cleaned sampled review data with the proxy helpfulness/business-value label.
- `sentiment_*.csv`: calibrated sentiment data built from rating plus VADER thresholds.
- `pseudo_summary_*.csv`: negative-review complaint-title pairs used for T5 fine-tuning.
- `*_manifest.json`: metadata files describing dataset rules, split sizes, and generation/re-split settings.

This folder is important because it captures the exact intermediate artifacts used to train the reported models.

### `models/`

- Stores lightweight evaluation artifacts and selected model bundles.
- `review_value_classifier.pkl`: saved scikit-learn review-value classifier.
- `*_metrics.json`: metrics used in the report and dashboard.
- `*_samples.json`: qualitative generation examples for the title-generation task.
- `zero_vs_base_3sample_comparison.json`: small qualitative comparison between zero-shot and fine-tuned title generation.

Full Transformer weight files and checkpoint directories are not committed because they are too large for a normal coursework repository. The repository keeps lightweight evaluation artifacts such as JSON metrics and sample outputs, and may also contain small tokenizer/config files. These lightweight files are not enough to replace the full trained model weights. The full local model folders can be recreated with the training commands below.

For full live inference in a fresh clone, recreate or copy these directories before using the `Single Review Check` and `Merchant Upload` tabs:

- `models/bert_sentiment/`: required by the BERT sentiment classifier. The app loads it with `local_files_only=True`.
- `models/t5_pseudo_summary_base/` or `models/t5_pseudo_summary_small/`: preferred local complaint-title generators.

Without these complete local Transformer folders, the saved metrics and processed datasets remain available for inspection, but BERT-based live sentiment inference will not run until `models/bert_sentiment/` is rebuilt. Complaint-title generation can fall back to Ark, cached zero-shot T5, or local rules depending on the runtime environment.

### `src/helpfulness/`

- Review-value dataset preparation and model training.
- [prepare_helpfulness_dataset.py](src/helpfulness/prepare_helpfulness_dataset.py): samples Amazon reviews, cleans them, and creates the proxy label.
- [train_helpfulness.py](src/helpfulness/train_helpfulness.py): trains the TF-IDF + Logistic Regression classifier.
- [predict_helpfulness.py](src/helpfulness/predict_helpfulness.py): inference helper for runtime use.

### `src/sentiment/`

- Sentiment dataset calibration and BERT fine-tuning.
- [label_calibration.py](src/sentiment/label_calibration.py): applies the rating + VADER rules and writes calibrated CSV splits.
- [train_bert_sentiment.py](src/sentiment/train_bert_sentiment.py): fine-tunes and evaluates `bert-base-uncased`.

### `src/summarization/`

- Complaint-title generation logic and T5 training.
- [generate_pseudo_titles.py](src/summarization/generate_pseudo_titles.py): uses Ark as the teacher model to create pseudo labels.
- [fine_tune_t5_pseudo.py](src/summarization/fine_tune_t5_pseudo.py): trains local Flan-T5 student models.
- [train_t5.py](src/summarization/train_t5.py): shared title-generation evaluation utilities, including ROUGE logic.
- [sample_zero_vs_base.py](src/summarization/sample_zero_vs_base.py): creates a small qualitative comparison artifact.

### `src/preprocessing/`

- Shared preprocessing entry points and helper wrappers.
- Includes cleaned text preparation and the thin command entry point used for complaint-title generation and re-splitting.

### `src/visualization/`

- Dashboard-side aggregation and retrieval helpers.
- [dashboard_utils.py](src/visualization/dashboard_utils.py) prepares scope-level metrics, representative review retrieval, and assistant-style responses.

### `src/utils/`

- Miscellaneous reusable helpers, including dataset-loading utilities.

## Why The Structure Matters

For a teacher or reviewer, the key takeaway is that the repository is separated into clear layers:

1. `data construction`
2. `model training`
3. `evaluation artifacts`
4. `user-facing dashboard`

That structure makes it easier to verify that the reported metrics, processed data, and application behavior all come from the same reproducible pipeline rather than from disconnected experiments.

## Environment Setup

Project was last run with Python `3.9.6`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run Dashboard

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

If Ark is enabled:

```bash
export ARK_API_KEY="your-ark-api-key"
export ARK_MODEL="doubao-seed-2-0-lite-260215"
export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
.venv/bin/streamlit run app/streamlit_app.py
```

Do not launch with `python app/streamlit_app.py`.

Current title model priority in app runtime:

1. `models/t5_pseudo_summary_base/`
2. `models/t5_pseudo_summary_small/`
3. `models/t5_pseudo_summary/` (legacy)
4. Ark fallback (if configured)
5. Zero-shot cached fallback
6. Local heuristic fallback

## End-To-End Rebuild Commands

Run from project root.

1. Build review-value dataset

```bash
.venv/bin/python -m src.helpfulness.prepare_helpfulness_dataset \
  --categories All_Beauty Amazon_Fashion Appliances Handmade_Products Health_and_Personal_Care \
  --samples-per-category 2000 \
  --min-helpful-per-category 250 \
  --helpful-vote-threshold 2 \
  --limit 20000
```

2. Calibrate sentiment labels

```bash
.venv/bin/python -m src.sentiment.label_calibration \
  --positive-rating-min 4 \
  --negative-rating-max 3 \
  --positive-score-threshold 0.05 \
  --negative-score-threshold -0.05
```

3. Generate pseudo complaint titles (teacher model)

```bash
.venv/bin/python -m src.preprocessing.generate_complaint_titles \
  --request-timeout 60
```

4. Re-split pseudo-title data (`70/10/20`)

```bash
.venv/bin/python -m src.preprocessing.resplit_complaint_titles
```

5. Train review-value model

```bash
.venv/bin/python -m src.helpfulness.train_helpfulness
```

6. Train sentiment model

```bash
.venv/bin/python -m src.sentiment.train_bert_sentiment
```

7. Train title generators

```bash
.venv/bin/python -m src.summarization.fine_tune_t5_pseudo \
  --model-name google/flan-t5-small \
  --output-dir models/t5_pseudo_summary_small \
  --metrics-output models/t5_pseudo_summary_small_metrics.json \
  --samples-output models/t5_pseudo_summary_small_samples.json \
  --allow-download \
  --bertscore-model-type distilbert-base-uncased
```

```bash
.venv/bin/python -m src.summarization.fine_tune_t5_pseudo \
  --model-name google/flan-t5-base \
  --output-dir models/t5_pseudo_summary_base \
  --metrics-output models/t5_pseudo_summary_base_metrics.json \
  --samples-output models/t5_pseudo_summary_base_samples.json \
  --allow-download \
  --bertscore-model-type distilbert-base-uncased
```

Optional PPT qualitative check:

```bash
.venv/bin/python -m src.summarization.sample_zero_vs_base \
  --sample-size 3 \
  --seed 42
```

If `google/flan-t5-small` is not cached locally, append `--allow-download`.

## Key Caveats (For Evaluation)

- `review_value_label` is a proxy (`helpful_votes >= 2`), not a human gold quality label.
- Sentiment labels are calibrated with rating + VADER compound thresholds (positive >= 0.05, negative <= -0.05), not manual annotations.
- Dashboard complaint candidates use a broader exploration pool (`calibrated negative` or `rating <= 3`) so product-level issue exploration is not too sparse; sentiment metrics still use the calibrated sentiment labels.
- Complaint-title targets are LLM-generated pseudo labels, not human-written gold titles.
- Title-generation results should be interpreted as demo-oriented effectiveness, not large-scale production validation.
