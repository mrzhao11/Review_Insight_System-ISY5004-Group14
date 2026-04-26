# Model Artifacts

This repository keeps lightweight model outputs such as metric JSON files and the small review-value classifier bundle.

Large transformer model directories are intentionally ignored by Git:

- `models/bert_sentiment/`
- `models/t5_summary/`
- `models/t5_pseudo_summary/`
- `models/t5_pseudo_summary_small/`
- `models/t5_pseudo_summary_base/`

They are available in the local working copy used for development, but they should not be pushed to a normal GitHub repository because the weights are several gigabytes.

To recreate them, run:

```bash
.venv/bin/python -m src.sentiment.train_bert_sentiment
.venv/bin/python -m src.preprocessing.generate_complaint_titles
.venv/bin/python -m src.preprocessing.resplit_complaint_titles
.venv/bin/python -m src.summarization.fine_tune_t5_pseudo \
  --model-name google/flan-t5-small \
  --output-dir models/t5_pseudo_summary_small \
  --metrics-output models/t5_pseudo_summary_small_metrics.json \
  --samples-output models/t5_pseudo_summary_small_samples.json \
  --allow-download \
  --bertscore-model-type distilbert-base-uncased
.venv/bin/python -m src.summarization.fine_tune_t5_pseudo \
  --model-name google/flan-t5-base \
  --output-dir models/t5_pseudo_summary_base \
  --metrics-output models/t5_pseudo_summary_base_metrics.json \
  --samples-output models/t5_pseudo_summary_base_samples.json \
  --allow-download \
  --bertscore-model-type distilbert-base-uncased
```

The complaint-title preprocessing command uses Ark as a teacher for calibrated negative reviews. The re-split command currently produces a 560/80/160 pseudo-title split, and the fine-tuning commands train local Flan-T5-small and Flan-T5-base students in `models/t5_pseudo_summary_small/` and `models/t5_pseudo_summary_base/`. The zero-shot summarization command evaluates the baseline on the same pseudo-title split and now records ROUGE-1, ROUGE-2, and ROUGE-L.

`src.preprocessing.generate_complaint_titles` requires `ARK_API_KEY`. If the Hugging Face checkpoints are already cached locally, `--allow-download` can be omitted from the T5 commands.
