# Review Insight System - Project Report (Draft)

## 1. Business Problem

Online merchants receive many customer reviews but usually lack a fast way to:

1. identify which reviews are business-relevant (high-value),
2. detect negative reviews that indicate actionable risk,
3. summarize long complaints into short issue titles for rapid decision-making.

This project builds an end-to-end review intelligence pipeline to support those needs.

## 2. Datasets Used

### 2.1 Raw Data Source

- Amazon Reviews 2023 dataset (McAuley-Lab):  
  [https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

### 2.2 Sampled Categories

- `All_Beauty`
- `Amazon_Fashion`
- `Appliances`
- `Handmade_Products`
- `Health_and_Personal_Care`

### 2.3 Processed Datasets in This Project

- Review-value dataset: 10,000 reviews (8,000 train / 1,000 validation / 1,000 test)
- Calibrated sentiment dataset: 3,820 train / 856 validation / 836 test
- Pseudo-title dataset (negative reviews with LLM-generated complaint titles):  
  833 train / 118 validation / 237 test

## 3. Solution Approaches

### 3.1 Review Value Classification

- Model: TF-IDF + Logistic Regression
- Goal: classify reviews into `high-value` vs `low-value`

### 3.2 Sentiment Classification

- Model: fine-tuned `bert-base-uncased`
- Labels: calibrated using rating + lexical sentiment signal
- Goal: classify `negative` vs `positive`

### 3.3 Complaint Title Generation

- Teacher labeling: LLM-generated pseudo complaint titles for negative reviews
- Student models: fine-tuned Flan-T5-small and Flan-T5-base
- Goal: generate concise complaint titles from review text

### 3.4 System Integration

- Streamlit dashboard for:
  - category/product scope filtering,
  - issue exploration,
  - single-review inference,
  - CSV upload analysis,
  - assistant-style Q&A.

## 4. Test Results

| Component | Main Test Results |
| --- | --- |
| Review value classifier | Accuracy 0.840; Positive-class F1 0.418 (test size 400 in training output) |
| Sentiment classifier (BERT) | Accuracy 0.982; F1 0.990; Macro-F1 0.939 |
| Complaint title model (Flan-T5-small, tuned) | ROUGE-1/2/L: 0.393 / 0.187 / 0.373; BERTScore F1 0.834 |
| Complaint title model (Flan-T5-base, tuned) | ROUGE-1/2/L: 0.435 / 0.230 / 0.412; BERTScore F1 0.845 |

Additional quick check for presentation:

- 3-sample zero-shot vs tuned base comparison (`seed=42`): base wins 3/3.

## 5. Conclusions

1. The end-to-end pipeline is functional and usable for business-facing review analysis.
2. Flan-T5-base is the best-performing complaint-title model among current candidates.
3. Sentiment model performance is strong on calibrated labels.
4. Review-value positive-class detection remains the main area for future improvement.

## 6. Public Libraries / Models / Source Code References

### Libraries

- Python
- pandas
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Streamlit
- Plotly
- joblib

### Public Models / External Resources

- `bert-base-uncased` (Hugging Face)
- `google/flan-t5-small` (Hugging Face)
- `google/flan-t5-base` (Hugging Face)
- Amazon Reviews 2023 dataset (McAuley-Lab)

### Project Source Code

- Repository:  
  [https://github.com/mrzhao11/Review_Insight_System-ISY5004-Group14](https://github.com/mrzhao11/Review_Insight_System-ISY5004-Group14)

## 7. Individual Contribution

This is a solo project.

I completed:

- data preparation and preprocessing,
- model training and evaluation,
- dashboard implementation and iteration,
- pipeline and project documentation.

Development was assisted by Codex for coding acceleration, debugging support, and documentation drafting/refinement.

## 8. Usage of GenAI / LLMs

### 8.1 Declaration

Yes, GenAI/LLM was used in this project.

### 8.2 Role and Tasks Performed

- generated pseudo complaint-title labels for negative reviews,
- supported coding/debugging and implementation refinement,
- supported report/documentation drafting and wording improvements.

### 8.3 Pros and Cons

**Pros**

- faster experimentation and implementation,
- improved productivity in repetitive engineering tasks,
- efficient pseudo-label generation for title summarization.

**Cons**

- LLM outputs still require manual validation and quality checks,
- potential inconsistency or hallucination without strict constraints,
- API-based generation introduces cost/dependency considerations.

