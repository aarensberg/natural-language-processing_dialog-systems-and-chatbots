# Chatbot Coursework

This repository contains the coursework chatbot pipeline in PyTorch. The current codebase is centered on the improved Exercise 2 model, while Exercise 1 artifacts are preserved for reference.

## Setup

1. Create a virtual environment and install dependencies.

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Download and extract the dialogue datasets. By default this downloads everything, but you can pass `--datasets` to choose only the corpora you need.

   ```bash
   python scripts/download_datasets.py
   ```

   Example:

   ```bash
   python scripts/download_datasets.py --datasets cornell personachat dailydialog
   ```

   Available values: `cornell`, `personachat`, `dailydialog`, `empatheticdialogues`, `opensubtitles`.

3. Check the dataset paths in [src/config.py](src/config.py). The defaults are:

   ```python
   CORNELL_MOVIE_DIR = DATA_DIR / "cornell movie-dialogs corpus"
   PERSONA_CHAT_DIR = DATA_DIR / "personachat"
   DAILY_DIALOG_DIR = DATA_DIR / "EMNLP_dataset"
   EMPATHETIC_DIALOGUES_DIR = DATA_DIR / "empatheticdialogues"
   OPEN_SUBTITLES_DIR = DATA_DIR / "opensubtitles"
   ```

## Artifact Layout

Keep only the exercise root folders in `artifacts/`:

```text
artifacts
├── exercise1
│   ├── checkpoints
│   └── outputs
├── exercise2
│   ├── checkpoints
│   └── outputs
├── exercise3
│   ├── checkpoints
│   └── outputs
└── exercise{4-7}
    ├── checkpoints
    └── outputs
```

Current exercise outputs are stored in:

- `artifacts/exercise1/outputs/`
- `artifacts/exercise2/outputs/`
- `artifacts/exercise2/outputs_no_glove/`
- `artifacts/exercise2/outputs_glove/`
- `artifacts/exercise3/outputs/`
- `artifacts/exercise4/outputs/`

## Running Exercise 2

The improved chatbot is the current executable pipeline.

```bash
python -m src.pipeline --epochs 5 --batch-size 128 --device cuda --use-glove --glove-path data/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt --output-dir artifacts/exercise2/outputs
```

Optional pretrained-embedding smoke test:

```bash
python -m src.pipeline --max-conversations 100 --epochs 1 --batch-size 16 --device cpu --use-glove --glove-path data/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt --output-dir artifacts/exercise2/outputs
```

Recommended Colab flow on a T4 GPU:

```bash
git clone https://github.com/aarensberg/natural-language-processing_dialog-systems-and-chatbots.git
cd natural-language-processing_dialog-systems-and-chatbots
pip install -r requirements.txt
python scripts/download_datasets.py --datasets cornell
curl -L "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.300d.zip" -o glove.2024.wikigiga.300d.zip
unzip glove.2024.wikigiga.300d.zip
rm glove.2024.wikigiga.300d.zip  # Optional cleanup
python -m src.pipeline --epochs 5 --batch-size 128 --device cuda --use-glove --glove-path data/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt --output-dir artifacts/exercise2/outputs
```

After the Colab run, copy these paths back into the local workspace:

- `artifacts/exercise2/checkpoints/seq2seq_attention.pt`
- `artifacts/exercise2/outputs/`
- `artifacts/exercise2/copilot-journal.md`

For the ablation comparison, keep both Colab result folders:

- `artifacts/exercise2/outputs_no_glove/`
- `artifacts/exercise2/outputs_glove/`

The GloVe run is the better reference for validation/test loss, while the no-GloVe run remains useful as the ablation baseline.

If you rerun a job, clear or archive the target `outputs/` directory first so the standardized layout stays clean.

## Running Exercise 3

Exercise 3 compares Cornell-only training with Cornell + PersonaChat training using the same attention-based seq2seq model and a corpus prefix in the source text.

The final mixed-domain run is now complete. Cornell-only remains the control baseline, while Cornell + PersonaChat is the final multi-dataset comparison. The current workspace contains the full metrics and generation artifacts for both branches, so no rerun is needed before moving to Exercise 4.

```bash
python -m src.pipeline_ex3 --experiments cornell_only cornell_plus_persona --epochs 5 --batch-size 128 --device cuda --use-glove --glove-path data/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt --output-dir artifacts/exercise3/outputs
```

The pipeline writes comparison outputs, domain-specific metrics, plots, and sample conversations under `artifacts/exercise3/outputs/{cornell_only,cornell_plus_persona}`. The final mixed-domain metrics are in `artifacts/exercise3/outputs/comparison.json`.

## Running Exercise 4

Exercise 4 adds a simple but defensible conversational memory layer on top of the existing chatbot. It uses a rule-based memory state for names, locations, preferences, and recent statements, and falls back to the seq2seq model when no direct memory answer is available.

```bash
python -m src.pipeline_ex4 --device cpu --output-dir artifacts/exercise4/outputs
```

The pipeline compares baseline responses with memory-aware responses on five test cases and writes the results to `artifacts/exercise4/outputs/memory_test_results.json` and `artifacts/exercise4/outputs/memory_test_report.txt`.

Recommended Colab flow on a T4 GPU:

```bash
git clone https://github.com/aarensberg/natural-language-processing_dialog-systems-and-chatbots.git
cd natural-language-processing_dialog-systems-and-chatbots
pip install -r requirements.txt
python scripts/download_datasets.py --datasets personachat
python -m src.pipeline_ex3 --experiments cornell_only cornell_plus_persona --epochs 5 --batch-size 128 --device cuda --use-glove --glove-path data/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt --output-dir artifacts/exercise3/outputs
```

## Datasets

The project supports these dialogue corpora:

1. Cornell Movie-Dialogs Corpus
2. PersonaChat
3. DailyDialog
4. EmpatheticDialogues
5. OpenSubtitles

See the dataset download links and references in the coursework brief if you need the sources again.
