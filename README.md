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
│   ├── outputs
│   └── copilot-journal.md
├── exercise2
│   ├── checkpoints
│   ├── outputs
│   └── copilot-journal.md
└── exercise{3-7}
    ├── checkpoints
    ├── outputs
    └── copilot-journal.md
```

Current exercise outputs are stored in:

- `artifacts/exercise1/outputs/`
- `artifacts/exercise1/checkpoints/`
- `artifacts/exercise2/outputs/`
- `artifacts/exercise2/checkpoints/`
- `artifacts/exercise2/outputs_no_glove/`
- `artifacts/exercise2/outputs_glove/`

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

## Datasets

The project supports these dialogue corpora:

1. Cornell Movie-Dialogs Corpus
2. PersonaChat
3. DailyDialog
4. EmpatheticDialogues
5. OpenSubtitles

See the dataset download links and references in the coursework brief if you need the sources again.
