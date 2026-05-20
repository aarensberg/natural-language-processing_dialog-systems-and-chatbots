# Usage

1. Create a virtual environment and install dependencies:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. Download and extract datasets using [download_datasets.py](scripts/download_datasets.py):

    ```bash
    python scripts/download_datasets.py
    ```

    At the end, you should have the following directory structure:

    ```text
    data
    ├── cornell movie-dialogs corpus
    │   ├── .DS_Store
    │   ├── chameleons.pdf
    │   ├── movie_characters_metadata.txt
    │   ├── movie_conversations.txt
    │   ├── movie_lines.txt
    │   ├── movie_titles_metadata.txt
    │   ├── raw_script_urls.txt
    │   └── README.txt
    ├── EMNLP_dataset
    │   ├── .DS_Store
    │   ├── dialogues_act.txt
    │   ├── dialogues_emotion.txt
    │   ├── dialogues_text.txt
    │   ├── dialogues_topic.txt
    │   ├── readme.txt
    │   ├── test
    │   │   ├── dialogues_act_test.txt
    │   │   ├── dialogues_emotion_test.txt
    │   │   └── dialogues_test.txt
    │   ├── train
    │   │   ├── dialogues_act_train.txt
    │   │   ├── dialogues_emotion_train.txt
    │   │   └── dialogues_train.txt
    │   └── validation
    │       ├── dialogues_act_validation.txt
    │       ├── dialogues_emotion_validation.txt
    │       └── dialogues_validation.txt
    ├── empatheticdialogues
    │   ├── test.csv
    │   ├── train.csv
    │   └── valid.csv
    ├── opensubtitles
    │   ├── LICENSE
    │   ├── OpenSubtitles.en-fr.en
    │   ├── OpenSubtitles.en-fr.fr
    │   └── README
    └── personachat
        ├── test_both_original.txt
        ├── test_both_revised.txt
        ├── test_none_original.txt
        ├── test_other_original.txt
        ├── test_other_revised.txt
        ├── test_self_original.txt
        ├── test_self_revised.txt
        ├── train_both_original.txt
        ├── train_both_revised.txt
        ├── train_none_original.txt
        ├── train_other_original.txt
        ├── train_other_revised.txt
        ├── train_self_original.txt
        ├── train_self_revised.txt
        ├── valid_both_original.txt
        ├── valid_both_revised.txt
        ├── valid_none_original.txt
        ├── valid_other_original.txt
        ├── valid_other_revised.txt
        ├── valid_self_original.txt
        └── valid_self_revised.txt

    9 directories, 51 files
    ```

3. Check each dataset directory path in [src/config.py](src/config.py) and update if necessary (should be correct by default).

    ```python
    # Define paths to each dataset directory (after extraction)
    CORNELL_MOVIE_DIR = DATA_DIR / "cornell movie-dialogs corpus"
    PERSONA_CHAT_DIR = DATA_DIR / "personachat"
    DAILY_DIALOG_DIR = DATA_DIR / "EMNLP_dataset"
    EMPATHETIC_DIALOGUES_DIR = DATA_DIR / "empatheticdialogues"
    OPEN_SUBTITLES_DIR = DATA_DIR / "opensubtitles"
    ```

# Exercise 1

The first exercise is implemented in PyTorch under `src/ex1/`. It contains:

- Cornell Movie-Dialogs preprocessing and conversation pair construction;
- exploratory statistics for the filtered dialogue pairs;
- a baseline GRU encoder-decoder chatbot;
- greedy decoding and beam search generation;
- training/validation/test split by conversation to reduce leakage;
- generated sample conversations written to `artifacts/exercise1/outputs/`.

Run the pipeline with:

```bash
python -m src.ex1.pipeline --epochs 3 --batch-size 64
```

For a faster smoke test, limit the number of conversations:

```bash
python -m src.ex1.pipeline --max-conversations 2000 --epochs 1 --batch-size 32
```

The main outputs are saved in:

- `artifacts/exercise1/outputs/eda_summary.json`
- `artifacts/exercise1/outputs/split_manifest.json`
- `artifacts/exercise1/outputs/training_history.json`
- `artifacts/exercise1/outputs/metrics.json`
- `artifacts/exercise1/outputs/sample_conversations.txt`
- `artifacts/exercise1/checkpoints/baseline_seq2seq.pt`

This baseline is intentionally simple so it can be extended later in the coursework with attention, improved decoding, memory, personality conditioning, and feedback.


# Datasets

1. **Cornell Movie-Dialogs Corpus**
    - [arXiv](https://arxiv.org/abs/1106.3077) — *Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs*
    - [Download Link](https://www.cs.cornell.edu/~cristian//data/cornell_movie_dialogs_corpus.zip) from [Cristian Danescu-Niculescu-Mizil](https://www.cs.cornell.edu/~cristian/research#paper-chameleons-in-imagined)

2. **PersonaChat**
    - [arXiv](https://arxiv.org/abs/1801.07243) — Personalizing Dialogue Agents: I have a dog, do you have pets too?
    - [Official Page](https://parl.ai/projects/personachat/)
    - [GitHub](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/personachat) — facebookresearch/ParlAI/parlai/tasks/personachat
    - [Download Link](http://parl.ai/downloads/personachat/personachat.tgz) given in [GitHub — build.py](https://github.com/facebookresearch/ParlAI/blob/main/parlai/tasks/personachat/build.py)

3. **DailyDialog**
    - [arXiv](https://arxiv.org/abs/1710.03957) — DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset
    - [Download Link](https://aclanthology.org/attachments/I17-1099.Datasets.zip) from [ACL Anthology](https://aclanthology.org/I17-1099/)

4. **EmpatheticDialogues**
    - [arXiv](https://arxiv.org/abs/1811.00207) — Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset
    - [GitHub](https://github.com/facebookresearch/EmpatheticDialogues) — facebookresearch/EmpatheticDialogues
    - [Download Link — dl.fbaipublicfiles.com](https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz)

5. **OpenSubtitles**
    - [Official Page](https://www.opensubtitles.org/fr)
    - [Download Link](https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-fr.txt.zip) from [OPUS](https://opus.nlpl.eu/corpora-search/en&fr)