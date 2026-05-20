from pathlib import Path
import urllib.request
import tarfile
import zipfile
import shutil

# Official dataset source URLs
CORNELL_MOVIE_URL = (
    "https://www.cs.cornell.edu/~cristian//data/cornell_movie_dialogs_corpus.zip"
)
PERSONA_CHAT_URL = "http://parl.ai/downloads/personachat/personachat.tgz"
DAILY_DIALOG_URL = "https://aclanthology.org/attachments/I17-1099.Datasets.zip"
EMPATHETIC_DIALOGUES_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
OPEN_SUBTITLES_URL = (
    "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-fr.txt.zip"
)


def download_file(url: str, dest_path: Path):
    """Download a file from a URL to a specified output path."""
    print(f"Downloading: {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print("✅ Download completed.")


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract a zip or tar.gz archive to a specified directory."""
    print(f"Extracting: {archive_path} -> {extract_to}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print("✅ Extraction completed.")
    elif archive_path.suffix in [".gz", ".tgz"]:
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)
        print("✅ Extraction completed.")
    else:
        print(f"❌ Unsupported archive format: {archive_path}")


if __name__ == "__main__":

    # ===== 0. Create output directory if it doesn't exist =====

    OUTPUT_DIR = Path("./data")
    OUTPUT_DIR.mkdir(exist_ok=True)

    #  ===== 1. Download and extract each dataset =====

    datasets = {
        "cornell_movie_dialogs_corpus.zip": CORNELL_MOVIE_URL,
        "personachat.tgz": PERSONA_CHAT_URL,
        "dailydialog.zip": DAILY_DIALOG_URL,
        "empatheticdialogues.tar.gz": EMPATHETIC_DIALOGUES_URL,
        "opensubtitles.zip": OPEN_SUBTITLES_URL,
    }

    for filename, url in datasets.items():
        dest_path = OUTPUT_DIR / filename
        download_file(url, dest_path)
        extract_archive(dest_path, OUTPUT_DIR)
        dest_path.unlink()  # Remove the original archive after extraction

    # ===== 2. Extract DailyDialogues sub-zip files =====

    dailydialog_dir = OUTPUT_DIR / "EMNLP_dataset"
    if dailydialog_dir.exists():
        sub_zip_files = dailydialog_dir.glob("*.zip")  # train, test and valid
        for sub_zip in sub_zip_files:
            extract_archive(sub_zip, dailydialog_dir)
            sub_zip.unlink()  # Remove the sub-zip file after extraction

    # ===== 3. Clean up the data/ directory =====

    # 3.1 Remove useless __MACOSX directory created by macOS when extracting zip files
    shutil.rmtree(OUTPUT_DIR / "__MACOSX", ignore_errors=True)

    # 3.2 Place OpenSubtitles files in a dedicated subdirectory
    opensubtitles_dir = OUTPUT_DIR / "opensubtitles"
    opensubtitles_dir.mkdir(exist_ok=True)

    for filename in [
        "LICENSE",
        "README",
        "OpenSubtitles.en-fr.en",
        "OpenSubtitles.en-fr.fr",
    ]:
        src_file = OUTPUT_DIR / filename
        if src_file.exists():
            shutil.move(src_file, opensubtitles_dir / filename)
