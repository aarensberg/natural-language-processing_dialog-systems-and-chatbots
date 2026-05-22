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
- `artifacts/exercise5/outputs/`

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

## Running Exercise 5

Exercise 5 uses a stable persona profile named Ari. The personality mechanism combines a fixed persona description, a prompt prefix that injects the profile into the model input, and rule-based answers for direct identity questions such as the name, role, interests, and adversarial attempts to change the persona.

```bash
python -m src.pipeline_ex5 --device cpu --output-dir artifacts/exercise5/outputs
```

The smoke test covers six cases, including repeated and adversarial questions. The final outputs live in `artifacts/exercise5/outputs/personality_test_results.json`, `artifacts/exercise5/outputs/personality_test_report.txt`, and `artifacts/exercise5/outputs/sample_conversations.txt`.

## Running Exercise 6

Exercise 6 implements a simple user feedback loop where corrections are stored and reapplied at inference time. The smoke test stores user-provided corrections in `artifacts/exercise6/outputs/feedback_store.json` and demonstrates before/after responses.

```bash
python -m src.pipeline_ex6 --output-dir artifacts/exercise6/outputs
```

When generating samples or evaluating outputs, the pipeline now consults the feedback store and replaces any generated response that matches a corrected query. The feedback artifacts produced by the smoke test are:

- `artifacts/exercise6/outputs/feedback_store.json`
- `artifacts/exercise6/outputs/feedback_before.json`
- `artifacts/exercise6/outputs/feedback_after.json`
- `artifacts/exercise6/outputs/feedback_results.json`
- `artifacts/exercise6/outputs/feedback_report.txt`
- `artifacts/exercise6/outputs/sample_before_after.txt`

To reproduce the integrated behaviour (model + feedback), run the normal generation pipeline (Exercise 2 or 3), then ensure `artifacts/exercise6/outputs/feedback_store.json` exists and is populated; the generation scripts will automatically apply stored corrections when producing `sample` or `generation_metrics` outputs.

### HTTP inference server (optional)

You can run a lightweight HTTP server that exposes the inference API and the feedback store. Install dependencies and start the server:

```bash
pip install -r requirements.txt
python -m src.server_infer
```

Endpoints:
- `POST /infer` with JSON `{ "query": "...", "method": "greedy|beam|sample" }` returns `{ "response": "..." }`.
- `POST /add_correction` with JSON `{ "query": "...", "correction": "..." }` saves a correction to the feedback store.
- `GET /feedback` returns the current feedback store as JSON.

### Generalization test (paraphrase coverage)

To measure how stored corrections generalise to paraphrases, run the extended feedback test which computes paraphrase coverage and writes `generalization_results.json` and `generalization_report.txt`:

```bash
python -m src.pipeline_ex6 --extended
```

The report contains per-paraphrase hits (direct correction found or matched by normalization) and an overall coverage metric.

The current lookup strategy uses a lightweight semantic score based on character similarity plus token overlap, so paraphrased requests such as "Could you tell me what food you like most?" can still resolve to the stored correction for "What is your favourite food?".

### Large evaluation protocol

To run the larger evaluation with before/after examples and error analysis, use:

```bash
python -m src.evaluate_feedback_ex6
```

This generates:

- `artifacts/exercise6/outputs/large_eval_results.json`
- `artifacts/exercise6/outputs/large_eval_report.txt`
- `artifacts/exercise6/outputs/large_eval_before_after.txt`
- `artifacts/exercise6/outputs/large_eval_failure_cases.json`

On the current run, the protocol reported positive paraphrase coverage of 0.882, a false-positive rate of 0.167 on negative controls, and overall accuracy of 0.870. The observed failure cases were concentrated in the self-description category and one unrelated control query that produced a false positive.

## Running Exercise 7

Exercise 7 aggregates the complete system evaluation, ablation study, and error analysis from the existing artifacts. It compares the baseline model, the improved attention-based model, and the memory, persona, and feedback variants.

```bash
python -m src.evaluate_ex7
```

The protocol writes the following files under `artifacts/exercise7/outputs/`:

- `ex7_summary.json`
- `ex7_report.txt`
- `ex7_examples.txt`

The summary covers baseline versus improved metrics, memory/persona/feedback accuracy, dataset and component ablations, and a five-case failure analysis focused on the feedback loop. The current run highlights the expected trend: the improved model lowers loss relative to the baseline, memory and persona achieve perfect scores on their targeted test sets, and feedback improves coverage but still shows a small false-positive risk on unrelated queries.

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
