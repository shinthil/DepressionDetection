# Depression Detection

A research project comparing **clinical interview data (DAIC-WOZ)** and **social media text (Reddit Mental Health)** for depression and mental-health risk detection. The work spans classical machine learning, multimodal feature fusion, LLM-based zero-shot and few-shot inference, prompt engineering, cross-domain transfer, and exploratory quantum-classifier comparisons.

> **Disclaimer:** This project is for academic and research purposes only. It is **not** a clinical diagnostic tool and must not be used to make medical or mental-health decisions.

---

## Overview

Mental-health screening traditionally relies on structured clinical interviews. This repository investigates whether similar signals can be extracted from **spoken clinical transcripts** and **Reddit posts**, and how different modeling approaches compare:

| Approach | Description |
|----------|-------------|
| **Classical ML** | SVM, Logistic Regression, and MLP on sentence embeddings |
| **Multimodal fusion** | Combined audio (Wav2Vec2) + text (MiniLM) features from DAIC-WOZ |
| **Zero-shot LLM** | BART-large-MNLI for label inference without fine-tuning |
| **Few-shot LLM** | Example-guided prompting vs. zero-shot |
| **Prompt engineering** | Simple, descriptive, and clinical prompt variants |
| **Cross-domain transfer** | Train on one domain (DAIC or Reddit), test on the other |

---

## Repository Structure

```
Depression-Detection/
├── major 1/                          # Part 1 — DAIC-WOZ (clinical interviews)
│   ├── preprocess_final.ipynb        # Transcript preprocessing
│   ├── featureFusion.py              # Audio + text multimodal fusion
│   ├── Fusion_Streaming_Safe_v2.ipynb
│   ├── Zero_shot_Testing_Depression_VSCode.ipynb
│   ├── Zero_Shot_Depression_GPT4_Integrated.ipynb
│   ├── Depression_Quantum_Comparison_Qiskit_Improved (1).ipynb
│   ├── Final_Depression_Detection_Project_Complete (1).ipynb
│   └── *.csv                         # AVEC 2017 splits and label files
│
├── major 2/                          # Part 2 — Reddit pipeline & experiments
│   ├── 01_reddit_preprocess_embed.ipynb
│   ├── 02_reddit_train_eval_visualize.ipynb
│   ├── 03_cross_domain_transfer_daic_reddit.ipynb
│   ├── 04_llm_setup_and_sampling.ipynb
│   ├── 05_zeroshot_vs_trained.ipynb
│   ├── 06_prompt_engineering.ipynb
│   ├── 07_fewshot_vs_zeroshot.ipynb
│   ├── 08_error_analysis.ipynb
│   ├── 09_combined_results.ipynb
│   ├── reddit_part2_pipeline.ipynb   # End-to-end Reddit pipeline
│   ├── processing.ipynb
│   └── pipeline_utils.py             # Shared utilities (labeling, CV, metrics)
│
└── outputs/                          # Generated experiment results
    ├── reddit_cv_summary.csv
    ├── combined_experiment_comparison.csv
    ├── transfer_summary.csv
    ├── zeroshot_basic_metrics.json
    ├── fewshot_vs_zeroshot.csv
    └── ...
```

---

## Datasets

This repository does **not** include raw datasets. You must obtain them separately:

### 1. DAIC-WOZ (Part 1)
- **Source:** [DAIC-WOZ Depression Database](https://dcapswoz.ict.usc.edu/)
- **Contents:** Clinical interview audio, transcripts, and PHQ-8 depression scores
- **Usage:** Transcript preprocessing, multimodal fusion, zero-shot GPT-4 experiments, quantum comparison

### 2. Reddit Mental Health (Part 2)
- **Source:** [Kaggle — Reddit Mental Health Dataset](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset)
- **Contents:** Posts from mental-health-related subreddits
- **Label mapping:**
  - **Label 0 — Mental Health Risk:** `depression`, `anxiety`, `lonely`, `mentalhealth`
  - **Label 1 — High Risk (Suicidal):** `suicidewatch`

---

## Setup

### Requirements

- Python 3.10+
- Jupyter Notebook / JupyterLab (or Google Colab)
- GPU recommended for embedding generation and Wav2Vec2 (optional but faster)

### Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install sentence-transformers transformers torch
pip install librosa tqdm
```

For quantum experiments (Part 1):

```bash
pip install qiskit
```

For GPT-4 zero-shot experiments, an OpenAI API key is required.

---

## How to Run

### Part 1 — DAIC-WOZ (Clinical)

1. Download and extract the DAIC-WOZ dataset.
2. Update file paths in the notebooks to point to your local DAIC-WOZ directory.
3. Run in order:
   - `preprocess_final.ipynb` — clean transcripts
   - `featureFusion.py` — fuse audio + text embeddings
   - `Zero_shot_Testing_Depression_VSCode.ipynb` — zero-shot evaluation
   - `Final_Depression_Detection_Project_Complete (1).ipynb` — full project notebook

```bash
python "major 1/featureFusion.py" \
  --audio_dir ./audio \
  --transcript_dir ./transcript \
  --label_csv "./major 1/avec_combined_labels (1).csv"
```

### Part 2 — Reddit Pipeline (Recommended order)

Run the numbered notebooks sequentially:

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `01_reddit_preprocess_embed.ipynb` | Load, label, clean, balance, embed |
| 2 | `02_reddit_train_eval_visualize.ipynb` | Train SVM / LR / MLP with 5-fold CV |
| 3 | `03_cross_domain_transfer_daic_reddit.ipynb` | Cross-domain transfer experiments |
| 4 | `04_llm_setup_and_sampling.ipynb` | Sample data for LLM experiments |
| 5 | `05_zeroshot_vs_trained.ipynb` | Compare zero-shot vs. trained models |
| 6 | `06_prompt_engineering.ipynb` | Test prompt variants |
| 7 | `07_fewshot_vs_zeroshot.ipynb` | Few-shot vs. zero-shot comparison |
| 8 | `08_error_analysis.ipynb` | Misclassification analysis |
| 9 | `09_combined_results.ipynb` | Unified results and summary |

Set `DATA_PATH` in notebook 01 to your Reddit CSV file or directory before running.

Alternatively, run the all-in-one pipeline:

```
major 2/reddit_part2_pipeline.ipynb
```

---

## Key Results

Results below are from the included `outputs/` artifacts (Reddit binary classification: Mental Health Risk vs. High Risk).

### Classical models (5-fold stratified CV on Reddit)

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| **SVM** | 78.9% | 78.9% |
| MLP | 76.4% | 76.4% |
| Logistic Regression | 75.7% | 75.7% |

### Zero-shot vs. trained (BART-large-MNLI)

| Method | Accuracy | Weighted F1 |
|--------|----------|-------------|
| SVM (trained) | 78.9% | 78.9% |
| BART zero-shot | 56.5% | 54.6% |

Trained classical models consistently outperform zero-shot LLM inference on this task.

### Prompt engineering

| Prompt variant | Accuracy | Weighted F1 |
|----------------|----------|-------------|
| **Prompt B (descriptive)** | 71.0% | 70.3% |
| Prompt A (simple) | 68.5% | 65.7% |
| Prompt C (clinical) | 53.5% | 41.4% |

### Few-shot vs. zero-shot

| Method | Accuracy | Weighted F1 |
|--------|----------|-------------|
| **Zero-shot** | 71.0% | 70.3% |
| Few-shot | 60.5% | 60.3% |

Zero-shot outperformed few-shot for this setup. Best label pair: *"general depression anxiety and loneliness"* vs. *"suicidal ideation and self-harm"*.

### Cross-domain transfer

| Direction | Best model | Accuracy | Weighted F1 |
|-----------|------------|----------|-------------|
| Reddit → DAIC | MLP | 57.8% | 58.7% |
| DAIC → Reddit | LR / MLP | ~50.5% | ~50.4% |

Cross-domain performance drops significantly, indicating domain shift between clinical speech and social media text.

---

## Models & Tools Used

| Component | Model / Library |
|-----------|-----------------|
| Text embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Zero-shot classification | `facebook/bart-large-mnli` |
| Audio embeddings | `facebook/wav2vec2-base-960h` |
| Classical classifiers | scikit-learn (SVM, LR, MLP) |
| Quantum experiments | Qiskit |
| LLM experiments | OpenAI GPT-4 (Part 1) |

---

## Outputs

All experiment artifacts are saved under `outputs/`:

- `reddit_cv_summary.csv` — cross-validation metrics for classical models
- `combined_experiment_comparison.csv` — unified comparison across all sub-experiments
- `transfer_summary.csv` — cross-domain transfer results
- `zeroshot_basic_metrics.json` — zero-shot BART metrics
- `fewshot_vs_zeroshot.csv` — few-shot vs. zero-shot comparison
- `error_analysis_full.csv` — per-sample error analysis
- `best_prompt.json` — best-performing prompt configuration

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

This project is intended for academic and research use. Please verify dataset licenses (DAIC-WOZ and Kaggle Reddit dataset) before redistribution or commercial use.

---

## Acknowledgements

- [DAIC-WOZ](https://dcapswoz.ict.usc.edu/) — USC Institute for Creative Technologies
- [Reddit Mental Health Dataset](https://www.kaggle.com/datasets/entenam/reddit-mental-health-dataset) — Kaggle
- [Sentence Transformers](https://www.sbert.net/) — UKP Lab, TU Darmstadt
- [Hugging Face Transformers](https://huggingface.co/transformers/) — BART, Wav2Vec2
