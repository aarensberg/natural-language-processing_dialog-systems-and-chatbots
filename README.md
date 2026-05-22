# Chatbot Coursework

This repository contains the final PyTorch implementation for the chatbot coursework. The project is organised around seven exercises that progressively add preprocessing, stronger modelling, multi-dataset training, memory, personality control, user feedback, and final evaluation.

## Overview

The codebase is intentionally split into small, inspectable components:

- Exercise 1 builds the Cornell Movie-Dialogs preprocessing pipeline, baseline seq2seq model, and initial evaluation artifacts.
- Exercise 2 adds a stronger attention-based chatbot and compares greedy, beam, and sampling-based decoding.
- Exercise 3 compares Cornell-only training against Cornell plus PersonaChat.
- Exercise 4 adds deterministic conversational memory.
- Exercise 5 adds a stable persona profile and adversarial consistency checks.
- Exercise 6 adds a persistent feedback store and applies it at inference time.
- Exercise 7 aggregates the full evaluation, ablation study, and error analysis.

The project is fully trainable in PyTorch. No pretrained generative chatbot API is used for response generation.

## Setup

1. Create a local virtual environment and install dependencies.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download the dialogue corpora you need. By default the downloader fetches all supported datasets, but you can restrict it with `--datasets`.

   ```bash
   python scripts/download_datasets.py
   ```

   Example:

   ```bash
   python scripts/download_datasets.py --datasets cornell personachat dailydialog
   ```

   Supported values are `cornell`, `personachat`, `dailydialog`, `empatheticdialogues`, and `opensubtitles`.

3. Confirm the dataset paths in [src/config.py](src/config.py). The default locations are:

   - [src/config.py](src/config.py)
   - [data/cornell movie-dialogs corpus](data/cornell%20movie-dialogs%20corpus)
   - [data/personachat](data/personachat)
   - [data/EMNLP_dataset](data/EMNLP_dataset)
   - [data/empatheticdialogues](data/empatheticdialogues)
   - [data/opensubtitles](data/opensubtitles)

## Artifact Layout

Keep the repository outputs grouped by exercise:

```text
artifacts/
  exercise1/
  exercise2/
  exercise3/
  exercise4/
  exercise5/
  exercise6/
  exercise7/
```

The current workspace already contains the validated outputs and checkpoints for Exercises 1 to 7. The most important locations are:

- [artifacts/exercise1/outputs](artifacts/exercise1/outputs)
- [artifacts/exercise1/checkpoints/baseline_seq2seq.pt](artifacts/exercise1/checkpoints/baseline_seq2seq.pt)
- [artifacts/exercise2/outputs](artifacts/exercise2/outputs)
- [artifacts/exercise2/outputs_no_glove](artifacts/exercise2/outputs_no_glove)
- [artifacts/exercise2/outputs_glove](artifacts/exercise2/outputs_glove)
- [artifacts/exercise2/checkpoints/seq2seq_attention.pt](artifacts/exercise2/checkpoints/seq2seq_attention.pt)
- [artifacts/exercise3/outputs](artifacts/exercise3/outputs)
- [artifacts/exercise3/checkpoints](artifacts/exercise3/checkpoints)
- [artifacts/exercise4/outputs](artifacts/exercise4/outputs)
- [artifacts/exercise5/outputs](artifacts/exercise5/outputs)
- [artifacts/exercise6/outputs](artifacts/exercise6/outputs)
- [artifacts/exercise7/outputs](artifacts/exercise7/outputs)

Exercises 4 to 6 intentionally do not store new model checkpoints because they reuse the existing neural checkpoints from earlier exercises and add deterministic wrappers on top.

## Exercise 1

Exercise 1 builds the Cornell baseline.

```bash
python -m src.pipeline --epochs 2 --batch-size 64 --device cpu --output-dir artifacts/exercise1/outputs
```

It produces the Cornell statistics, plots, training history, sample conversations, vocabulary files, and the baseline checkpoint in [artifacts/exercise1/checkpoints/baseline_seq2seq.pt](artifacts/exercise1/checkpoints/baseline_seq2seq.pt).

## Exercise 2

Exercise 2 adds attention, a stronger GRU encoder-decoder, optional GloVe initialisation, and multiple decoding strategies.

```bash
python -m src.pipeline --epochs 5 --batch-size 128 --device cuda --use-glove --glove-path data/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt --output-dir artifacts/exercise2/outputs
```

Optional smoke test:

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
rm glove.2024.wikigiga.300d.zip
python -m src.pipeline --epochs 5 --batch-size 128 --device cuda --use-glove --glove-path data/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt --output-dir artifacts/exercise2/outputs
```

The no-GloVe and GloVe outputs are preserved separately for the ablation comparison. The checkpoint is [artifacts/exercise2/checkpoints/seq2seq_attention.pt](artifacts/exercise2/checkpoints/seq2seq_attention.pt).

## Exercise 3

Exercise 3 compares Cornell-only training against Cornell plus PersonaChat training using the same attention-based seq2seq architecture.

```bash
python -m src.pipeline_ex3 --experiments cornell_only cornell_plus_persona --epochs 5 --batch-size 128 --device cuda --use-glove --glove-path data/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt --output-dir artifacts/exercise3/outputs
```

The pipeline writes domain-specific outputs, sample conversations, plots, and the final comparison summary under [artifacts/exercise3/outputs](artifacts/exercise3/outputs). The final mixed-domain metrics are stored in [artifacts/exercise3/outputs/comparison.json](artifacts/exercise3/outputs/comparison.json).

## Exercise 4

Exercise 4 adds a deterministic conversational memory layer for names, locations, preferences, and recent statements.

```bash
python -m src.pipeline_ex4 --device cpu --output-dir artifacts/exercise4/outputs
```

The memory layer is rule-based by design, which makes the targeted recall tests predictable and easy to inspect. The pipeline also includes a small vector-based generalisation pass for paraphrased recall questions, plus a simple safety refusal for clearly unsafe prompts.

Generated artifacts:

- [artifacts/exercise4/outputs/memory_test_results.json](artifacts/exercise4/outputs/memory_test_results.json)
- [artifacts/exercise4/outputs/memory_test_report.txt](artifacts/exercise4/outputs/memory_test_report.txt)
- [artifacts/exercise4/outputs/memory_vector_test_results.json](artifacts/exercise4/outputs/memory_vector_test_results.json)
- [artifacts/exercise4/outputs/memory_vector_test_report.txt](artifacts/exercise4/outputs/memory_vector_test_report.txt)

## Exercise 5

Exercise 5 adds a stable persona profile named Ari.

```bash
python -m src.pipeline_ex5 --device cpu --output-dir artifacts/exercise5/outputs
```

The personality mechanism combines a fixed profile, a persona-prefixed prompt, and rule-based overrides for direct identity questions and adversarial attempts to change the character. The smoke test covers six cases and writes the outputs to [artifacts/exercise5/outputs](artifacts/exercise5/outputs).

## Exercise 6

Exercise 6 implements a persistent feedback loop.

```bash
python -m src.pipeline_ex6 --output-dir artifacts/exercise6/outputs
```

Corrections are stored in [artifacts/exercise6/outputs/feedback_store.json](artifacts/exercise6/outputs/feedback_store.json) and automatically applied at inference time through the CLI and HTTP server. The feedback store is deterministic and JSON-backed, so the before/after results can be reproduced exactly.

Artifacts written by the smoke test and extended evaluation:

- [artifacts/exercise6/outputs/feedback_before.json](artifacts/exercise6/outputs/feedback_before.json)
- [artifacts/exercise6/outputs/feedback_after.json](artifacts/exercise6/outputs/feedback_after.json)
- [artifacts/exercise6/outputs/feedback_results.json](artifacts/exercise6/outputs/feedback_results.json)
- [artifacts/exercise6/outputs/feedback_report.txt](artifacts/exercise6/outputs/feedback_report.txt)
- [artifacts/exercise6/outputs/sample_before_after.txt](artifacts/exercise6/outputs/sample_before_after.txt)
- [artifacts/exercise6/outputs/generalization_results.json](artifacts/exercise6/outputs/generalization_results.json)
- [artifacts/exercise6/outputs/generalization_report.txt](artifacts/exercise6/outputs/generalization_report.txt)
- [artifacts/exercise6/outputs/large_eval_results.json](artifacts/exercise6/outputs/large_eval_results.json)
- [artifacts/exercise6/outputs/large_eval_report.txt](artifacts/exercise6/outputs/large_eval_report.txt)
- [artifacts/exercise6/outputs/large_eval_before_after.txt](artifacts/exercise6/outputs/large_eval_before_after.txt)
- [artifacts/exercise6/outputs/large_eval_failure_cases.json](artifacts/exercise6/outputs/large_eval_failure_cases.json)

### HTTP inference server

```bash
python -m src.server_infer
```

Endpoints:

- `POST /infer` with JSON `{ "query": "...", "method": "greedy|beam|sample" }`
- `POST /add_correction` with JSON `{ "query": "...", "correction": "..." }`
- `GET /feedback`

### Transparency note

Some evaluation scripts use controlled fixtures so the behaviour of the new component can be measured directly. In particular, Exercise 6 seeds canonical corrections before the paraphrase and generalization checks, which is intentional and documented. This is not model leakage: it is a test harness for the feedback mechanism.

The semantic lookup used by the feedback store is lightweight and rule-based. It combines character similarity and token overlap instead of a neural retriever, which keeps the pipeline reproducible and transparent.

## Exercise 7

Exercise 7 aggregates the complete system evaluation, ablation study, and error analysis.

```bash
python -m src.evaluate_ex7
```

It writes:

- [artifacts/exercise7/outputs/ex7_summary.json](artifacts/exercise7/outputs/ex7_summary.json)
- [artifacts/exercise7/outputs/ex7_report.txt](artifacts/exercise7/outputs/ex7_report.txt)
- [artifacts/exercise7/outputs/ex7_examples.txt](artifacts/exercise7/outputs/ex7_examples.txt)

The summary compares baseline versus improved metrics, memory/persona/feedback behaviour, and the main failure modes of the feedback loop. The report is a critical evaluation of the whole system rather than a new training run.

## Checkpoint Fallbacks

Exercises 4 to 6 are intentionally wrapper-based. They reuse the earlier neural checkpoints when needed:

- Exercise 4 first looks for an Exercise 4 checkpoint, then falls back to Exercise 3, then Exercise 2.
- Exercise 5 first looks for an Exercise 5 checkpoint, then falls back to Exercise 4, then Exercise 3.
- Exercise 6 first looks for an Exercise 6 checkpoint, then falls back to Exercise 4, then Exercise 3, then Exercise 2.

This keeps the memory, persona, and feedback experiments reproducible without requiring redundant retraining.

## Datasets

The project supports these dialogue corpora:

1. Cornell Movie-Dialogs Corpus
2. PersonaChat
3. DailyDialog
4. EmpatheticDialogues
5. OpenSubtitles

The coursework runs in this repository primarily rely on Cornell and PersonaChat, with the other corpora kept available for extensions and comparison.
