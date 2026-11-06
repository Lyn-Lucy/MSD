# Multimodal Speculative Decoding (MSD)

📄 [**Paper on arXiv**](https://arxiv.org/pdf/2505.14260)

*Speculative Decoding Reimagined for Multimodal Large Language Models*

## 🧠 MSD Models

The following repositories contain MSD **draft checkpoints**, to be used with
these target models and the implementation in this repository.

| Target model | MSD draft checkpoint | Evaluation adapter |
|---|---|---|
| [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | [lucylyn/MSD-LLaVA1.5-7B](https://huggingface.co/lucylyn/MSD-LLaVA1.5-7B) | `llava_hf_msd` |
| [LLaVA-1.6-7B](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) | [lucylyn/MSD-LLaVA1.6-7B](https://huggingface.co/lucylyn/MSD-LLaVA1.6-7B) | `llava_hf_msd` |
| [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | [lucylyn/MSD-Qwen2.5VL-7B-Instruct](https://huggingface.co/lucylyn/MSD-Qwen2.5VL-7B-Instruct) | `qwen2_vl_msd` |

Each draft repository contains `config.json` and `pytorch_model.bin`.
`--msd_model_path` accepts either a Hugging Face repository ID or a local
checkpoint directory. MSD uses **KMeans-32 by default** for the draft model's
visual input; the target model keeps the complete visual input.

## 🧱 1. Setup & Installation

Use Python 3.10 and a CUDA 12.1-compatible environment. From the repository root:

```bash
conda create -n msd python=3.10 -y
conda activate msd

python -m pip install -e ./LLaVA
python -m pip install -e ./EAGLE
python -m pip install -e ./lmms-eval
```

## 📥 2. Download Datasets

Download the annotations used for instruction tuning:

* [`ShareGPT_V3_unfiltered_cleaned_split.json`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json)
* [`llava_v1_5_mix665k.json`](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

  > ⚠️ Before use, process `llava_v1_5_mix665k.json` with [`EAGLE/eagle/ge_data/convert.py`](EAGLE/eagle/ge_data/convert.py) to fix formatting issues.

Then download the image data from the following datasets:

* **COCO**: [train2017](http://images.cocodataset.org/zips/train2017.zip)
* **GQA**: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
* **OCR-VQA**: [Download script (Google Drive)](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing)

  > 💡 Make sure all OCR-VQA images are saved as `.jpg`
* **TextVQA**: [train\_val\_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
* **Visual Genome**: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading, organize the data under `./image_data` in the following structure:

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

---

## ⚙️ 3. Data Processing

Generate hidden-state caches for draft-model training. For LLaVA-1.5 or
Qwen2.5-VL, use the allocator from the repository root:

```bash
python -m eagle.ge_data.allocation \
    --model_type <model_type> \
    --model <target_model_path> \
    --image_data_path <image_data_dir> \
    --json_data_path <annotation_file> \
    --outdir <output_data_dir> \
    --gpus 0
```

Use `llava_v15_v` / `llava_v15_t` for LLaVA-1.5. The existing
`qwen2_vl_v` / `qwen2_vl_t` identifiers select the Qwen2.5-VL generator.
`--gpus` accepts multiple GPU indices, and `--start` / `--end` select an
annotation range with an exclusive end index.

For LLaVA-1.6, use the configurable cache generator:

```bash
python -m eagle.ge_data.prepare_llava16_hidden_states \
    --model <target_model_path> \
    --annotation-path <annotation_file> \
    --image-root <image_data_dir> \
    --output-dir <output_data_dir> \
    --mode teacher
```

Use `--mode generate` to cache model-generated answers instead. The optional
`prepare_data_llava16.sh` and `prepare_data_qwen25vl.sh` scripts accept
`BASE_MODEL`, `IMAGE_DATA`, `JSON_DATA`, and `DATA_OUTPUT` as environment variables.

## 🏋️ 4. Train the Model

Train with DeepSpeed, supplying your own model, cache, and output paths:

```bash
cd EAGLE/eagle/train

deepspeed --master_port 29504 --include localhost:0 main_deepspeed.py \
    --deepspeed_config ds_config.json \
    --tmpdir_v <visual_data_path> \
    --tmpdir_t <text_data_path> \
    --basepath <target_model_path> \
    --cpdir <checkpoint_output_dir> \
    --config <draft_model_config>
```

For LLaVA-1.6 curriculum training, run from the repository root:

```bash
torchrun --standalone --nproc_per_node=1 \
    EAGLE/eagle/train/main_deepspeed_llava16_decouple_kmeans_repro.py \
    --basepath <target_model_path> \
    --tmpdir_v <visual_data_path> \
    --tmpdir_t <text_data_path> \
    --config EAGLE/eagle/train/llava_hf_v16_7B_config.json \
    --deepspeed_config EAGLE/eagle/train/ds_config_llava16_repro.json \
    --visual_tokens 32 \
    --cpdir <checkpoint_output_dir>
```

The curriculum trainer defaults to 30 epochs: 10 text-only epochs, 10 transition
epochs, then 10 multimodal epochs. Use `--num_epochs`, `--text_epochs`, and
`--transition_epochs` to change the schedule. Increase `--nproc_per_node` to use
more local GPUs.

The optional `ds_train.sh` launcher accepts `BASE_MODEL`, `VISUAL_DATA`,
`TEXT_DATA`, and `MODEL_CONFIG`; use `CKPT_DIR`, `GPU_INCLUDE`, and
`DEEPSPEED_CONFIG` to override its output, GPU, and DeepSpeed settings.

## 📊 5. Evaluate the Model

Run evaluation directly with `accelerate launch -m lmms_eval`.

### LLaVA-1.6-7B

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29506 -m lmms_eval \
    --model llava_hf_msd \
    --model_args pretrained=llava-hf/llava-v1.6-vicuna-7b-hf \
    --msd_model_path lucylyn/MSD-LLaVA1.6-7B \
    --tasks chartqa \
    --batch_size 1 \
    --limit 100 \
    --gen_kwargs temperature=0,max_new_tokens=512 \
    --use_msd
```

For LLaVA-1.5-7B, keep `--model llava_hf_msd` and replace the target model and
draft checkpoint with `llava-hf/llava-1.5-7b-hf` and
`lucylyn/MSD-LLaVA1.5-7B`, respectively.

### Qwen2.5-VL-7B

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29506 -m lmms_eval \
    --model qwen2_vl_msd \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
    --msd_model_path lucylyn/MSD-Qwen2.5VL-7B-Instruct \
    --tasks chartqa \
    --batch_size 1 \
    --limit 100 \
    --gen_kwargs temperature=0,max_new_tokens=512 \
    --use_msd
```

### Evaluation settings

- `--use_msd` enables speculative decoding. Omit it for standard decoding.
- KMeans visual compression is enabled by default for MSD; no
  `--visual_compression` flag is needed. Use `--visual_tokens` to change the
  default count of 32. Padding is off by default.
- Set `max_new_tokens` explicitly when comparing runs: ChartQA's task default
  is 16, while the examples above use a 512-token limit.
- Select benchmarks with `--tasks`, optionally comma-separated. Use
  `--output_path <directory>` to save results.
- The evaluator reports generation time and average accepted length directly.
  Timing includes request processing and excludes model loading. Stop competing
  GPU workloads before measuring speed. Measure standard decoding on the same
  hardware and samples before calculating a speedup.

The evaluation subsets used in our local runs are:

| Task | `--limit` | Requests |
|---|---:|---:|
| `vqav2` | 50 | 100 across validation and test |
| `ai2d` | 100 | 100 |
| `scienceqa_img` | 100 | 100 |
| `chartqa` | 100 | 100 |
| `textvqa_val` | 50 | 50 |
| `hallusion_bench_image` | 100 | 100 |
| `mmbench_en_test` | 100 | 100 |

For offline datasets saved with Hugging Face `save_to_disk`, optionally set
`MSD_EVAL_DATA_ROOT` to a directory containing subdirectories named after these
tasks. Otherwise, datasets load through the normal Hugging Face path.

## Affiliation

1. Shanghai Innovation Institute
2. Xiamen University
