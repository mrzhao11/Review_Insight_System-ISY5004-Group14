# Model Artifacts

This repository keeps lightweight model outputs such as metric JSON files and the small review-value classifier bundle.

Large transformer model directories are intentionally ignored by Git:

- `models/bert_sentiment/`
- `models/t5_summary/`
- `models/t5_pseudo_summary/`

They are available in the local working copy used for development, but they should not be pushed to a normal GitHub repository because the weights are several gigabytes.

To recreate them, run:

```bash
.venv/bin/python -m src.sentiment.train_bert_sentiment
.venv/bin/python -m src.summarization.train_t5 --allow-download
.venv/bin/python -m src.preprocessing.generate_complaint_titles
.venv/bin/python -m src.summarization.fine_tune_t5_pseudo --allow-download
```

The zero-shot summarization command evaluates the Flan-T5 baseline. The complaint-title preprocessing command uses Ark as a teacher for calibrated negative reviews, then the fine-tuning command trains a local T5 student in `models/t5_pseudo_summary/`.

`src.preprocessing.generate_complaint_titles` requires `ARK_API_KEY`. If `google/flan-t5-small` is already cached locally, `--allow-download` can be omitted from the T5 commands.
