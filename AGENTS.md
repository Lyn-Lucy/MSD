# Repository Guidelines

## Project Structure & Module Organization

This repository implements Multimodal Speculative Decoding (MSD). Core draft-model and cache code lives in `EAGLE/eagle/model/`; data generation is in `EAGLE/eagle/ge_data/`, and DeepSpeed training is in `EAGLE/eagle/train/`. `LLaVA/llava/` contains the adapted LLaVA base implementation. Evaluation code, model adapters, and task YAML files live under `lmms-eval/lmms_eval/`. Root shell scripts orchestrate preparation, training, and evaluation. Treat `ckpt/`, `logs*/`, `output/`, and `trainout/` as generated artifacts; `figure/` contains documentation assets.

## Build, Test, and Development Commands

Use Python 3.10 and a CUDA 12.1-compatible environment. From the repository root, install all three packages in editable mode:

```bash
python -m pip install -e ./LLaVA
python -m pip install -e ./EAGLE
python -m pip install -e ./lmms-eval
```

Generate data with `python -m eagle.ge_data.allocation --outdir <dir> --model_type <type> --model <path> --image_data_path <dir> --json_data_path <file>`. Train with `bash ds_train.sh`, after replacing its machine-specific paths and GPU list. Run a small evaluation before a full sweep, for example `accelerate launch -m lmms_eval --model llava_hf_msd --model_args pretrained=<path> --msd_model_path <path> --tasks chartqa --batch_size 1 --limit 10 --use_msd`.

## Coding Style & Naming Conventions

Use four-space indentation and conventional Python naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Keep task identifiers and YAML filenames aligned (for example, `mathvista_test.yaml`). Within `lmms-eval`, format with Black (240-character line length) and isort:

```bash
cd lmms-eval && pre-commit run --all-files
```

## Testing Guidelines

There is no repository-wide coverage gate. Add focused `test_*.py` tests near the affected package and run `python -m pytest <path>` when applicable. For model or task changes, run at least one `lmms_eval` smoke evaluation with `--limit 10`; report the task, model, GPU, and key metric. Use `python -m compileall EAGLE/eagle LLaVA/llava lmms-eval/lmms_eval` for a fast syntax check.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lowercase subjects such as `prepare data`; keep each commit scoped to one change. Before opening a PR, search for related issues, use a descriptive title, explain behavior and configuration changes, and list validation commands. Link relevant issues and attach metric excerpts or screenshots when results or UI behavior change. Do not commit checkpoints, datasets, cache files, logs, access tokens, or machine-specific absolute paths.
