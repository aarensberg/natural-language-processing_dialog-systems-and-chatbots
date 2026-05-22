from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EX1_DIR = ARTIFACTS_DIR / "exercise1"
EX1_CHECKPOINT_DIR = EX1_DIR / "checkpoints"
EX1_OUTPUT_DIR = EX1_DIR / "outputs"
EX1_REPORT_DIR = EX1_DIR / "report"
EX2_DIR = ARTIFACTS_DIR / "exercise2"
EX2_CHECKPOINT_DIR = EX2_DIR / "checkpoints"
EX2_OUTPUT_DIR = EX2_DIR / "outputs"
EX2_REPORT_DIR = EX2_DIR / "report"
EX3_DIR = ARTIFACTS_DIR / "exercise3"
EX3_CHECKPOINT_DIR = EX3_DIR / "checkpoints"
EX3_OUTPUT_DIR = EX3_DIR / "outputs"
EX3_REPORT_DIR = EX3_DIR / "report"
EX4_DIR = ARTIFACTS_DIR / "exercise4"
EX4_CHECKPOINT_DIR = EX4_DIR / "checkpoints"
EX4_OUTPUT_DIR = EX4_DIR / "outputs"
EX4_REPORT_DIR = EX4_DIR / "report"
EX5_DIR = ARTIFACTS_DIR / "exercise5"
EX5_CHECKPOINT_DIR = EX5_DIR / "checkpoints"
EX5_OUTPUT_DIR = EX5_DIR / "outputs"
EX5_REPORT_DIR = EX5_DIR / "report"

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
MODEL_NUM_LAYERS = 2
USE_BIDIRECTIONAL_ENCODER = True

USE_GLOVE = True
GLOVE_DIM = 300
GLOVE_FILE = (
    DATA_DIR
    / "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
)
FREEZE_GLOVE = False

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
