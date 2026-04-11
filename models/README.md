# Model Artifacts

This repository keeps lightweight model outputs such as metric JSON files and the small review-value classifier bundle.

Large transformer model directories are intentionally ignored by Git:

- `models/bert_sentiment/`
- `models/t5_summary/`

They are available in the local working copy used for development, but they should not be pushed to a normal GitHub repository because the weights are several gigabytes.

To recreate them, run:

```bash
.venv/bin/python -m src.sentiment.train_bert_sentiment
.venv/bin/python -m src.summarization.train_t5 --allow-download
```

The summarization command evaluates the zero-shot Flan-T5 baseline and writes the local model/cache-dependent outputs used by the demo.
