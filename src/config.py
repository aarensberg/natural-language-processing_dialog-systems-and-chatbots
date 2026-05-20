from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EX1_DIR = ARTIFACTS_DIR / "exercise1"
EX1_CHECKPOINT_DIR = EX1_DIR / "checkpoints"
EX1_OUTPUT_DIR = EX1_DIR / "outputs"
EX1_REPORT_DIR = EX1_DIR / "report"

assert (
    DATA_DIR.exists()
), f"Data directory {DATA_DIR} does not exist. Please run scripts/download_datasets.py to download the datasets."

# Define paths to each dataset directory (after extraction)
CORNELL_MOVIE_DIR = DATA_DIR / "cornell movie-dialogs corpus"
PERSONA_CHAT_DIR = DATA_DIR / "personachat"
DAILY_DIALOG_DIR = DATA_DIR / "EMNLP_dataset"
EMPATHETIC_DIALOGUES_DIR = DATA_DIR / "empatheticdialogues"
OPEN_SUBTITLES_DIR = DATA_DIR / "opensubtitles"

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

MIN_TOKEN_FREQ = 3
MAX_VOCAB_SIZE = 20000
MAX_SOURCE_TOKENS = 20
MAX_TARGET_TOKENS = 20

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

EMBED_DIM = 256
HIDDEN_DIM = 384
DROPOUT = 0.2
TEACHER_FORCING_RATIO = 0.55

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
