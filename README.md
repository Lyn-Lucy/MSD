# Multimodal Speculative Decoding (MSD)

📄 [**Paper on arXiv**](https://arxiv.org/pdf/2505.14260)
*Speculative Decoding Reimagined for Multimodal Large Language Models*
---

## 🧠 MSD Models

You can directly use the Multimodal Speculative Decoding (MSD) models available on Hugging Face:

- **MSD-LLaVA1.5-7B**: [lucylyn/MSD-LLaVA1.5-7B](https://huggingface.co/lucylyn/MSD-LLaVA1.5-7B)
- **MSD-LLaVA1.5-13B**: [lucylyn/MSD-LLaVA1.5-13B](https://huggingface.co/lucylyn/MSD-LLaVA1.5-13B)
- **MSD-Qwen2VL-7B-Instruct**: [lucylyn/MSD-Qwen2VL-7B-Instruct](https://huggingface.co/lucylyn/MSD-Qwen2VL-7B-Instruct)


---

## 🧱 1. Setup & Installation

```bash
conda create -n msd python=3.10 -y
conda activate msd
# Ensure CUDA 12.1 is installed and configured

cd LLaVA
pip install -e .
cd ../EAGLE
pip install -e .
cd ../lmms-eval
pip install -e .
```

---

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

Use the following script to generate training data. You can control the target model by setting the `--model_type` argument (e.g., `llava_v15_t/v` or `qwen2_vl_t/v`):

```bash
cd EAGLE/eagle/ge_data

CUDA_VISIBLE_DEVICES=0 python -m eagle.ge_data.allocation \
    --outdir <output_data_dir> \
    --model_type <model_type> \
    --model <base_model_path> \
    --image_data_path <image_data_dir> \
    --json_data_path <annotation_file>
```

---

## 🏋️ 4. Train the Model

Use DeepSpeed to train the speculative decoding model. Modify the following paths according to your setup:

```bash
cd EAGLE/eagle/train

deepspeed --master_port 29504 --include localhost:0 main_deepspeed.py \
    --deepspeed_config ds_config.json \
    --tmpdir_v <visual_data_path> \
    --tmpdir_t <text_data_path> \
    --basepath <base_llm_path> \
    --cpdir <checkpoint_output_dir> \
    --config <training_config_path>
```

**Parameters:**

* `<visual_data_path>`: directory containing preprocessed visual data
* `<text_data_path>`: directory containing preprocessed text data
* `<training_config_path>`: training configuration file, e.g., `llava_v15_7B_config.json`

---

## 📊 5. Evaluate the Model

Run evaluation with `lmms-eval`. The following example evaluates on the `ChartQA` task:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=29506 -m lmms_eval \
    --model <model_name> \
    --model_args pretrained="<base_model_path>" \
    --msd_model_path <msd_model_path> \
    --tasks chartqa \
    --batch_size 1 \
    --gen_kwargs temperature=0,max_new_tokens=512 \
    --limit 100 \
    --use_msd
```

**Parameters:**

* `<model_name>`: short name identifier of your model, e.g., `llava_hf_msd` or `qwen2_vl_msd`
* `<base_model_path>`: path to the base pretrained model
* `<msd_model_path>`: path to the MSD model
---

## Maintained training and evaluation entrypoints

Use `llava_hf_msd` for Hugging Face LLaVA/LLaVA-NeXT models and
`qwen2_vl_msd` for Qwen2.5-VL. MSD uses KMeans by default and retains 32
visual tokens; change the count with `--visual_tokens`. Standard decoding
is available by omitting `--use_msd`. Generation results retain the original
prompt prefix. The evaluator reports request-loop time and average accepted
length directly, without an auxiliary runner.

The LLaVA adapter uses a total sequence limit of 4096. Generation respects
task defaults and `--gen_kwargs`; specify `max_new_tokens=512` to reproduce
the previous LLaVA timing configuration. Timing excludes model loading.
Stop competing GPU workloads before measuring speed.

For offline datasets saved with Hugging Face `save_to_disk`, optionally set
`MSD_EVAL_DATA_ROOT` to a directory with task subdirectories:
`chartqa`, `textvqa_val`, `vqav2`, `ai2d`, `scienceqa_img`,
`hallusion_bench_image`, and `mmbench_en_test`. Otherwise datasets load
through the normal Hugging Face path.

Run evaluation with the direct `accelerate launch -m lmms_eval` command above.
Select one or more benchmarks with `--tasks` (comma-separated), and set
`--limit`, `--gen_kwargs`, and optional `--output_path` explicitly.

`ds_train.sh` accepts `BASE_MODEL`, `VISUAL_DATA`, `TEXT_DATA`,
`MODEL_CONFIG`, and optional `CKPT_DIR`, `GPU_INCLUDE`, and
`DEEPSPEED_CONFIG`. For LLaVA-NeXT curriculum training, invoke the trainer
directly:

```bash
torchrun --standalone --nproc_per_node=1 \
    EAGLE/eagle/train/main_deepspeed_llava16_decouple_kmeans_repro.py \
    --basepath <base_model_path> \
    --tmpdir_v <visual_data_path> --tmpdir_t <text_data_path> \
    --config EAGLE/eagle/train/llava_hf_v16_7B_config.json \
    --deepspeed_config EAGLE/eagle/train/ds_config_llava16_repro.json \
    --cpdir <checkpoint_output_dir>
```

Data generation uses `prepare_data_llava16.sh` or
`prepare_data_qwen25vl.sh`, with `BASE_MODEL`, `IMAGE_DATA`, `JSON_DATA`,
and `DATA_OUTPUT` set explicitly. Source code and configuration belong in
Git; model weights, datasets, logs, local editor settings, and generated
figures remain local.

## Affiliation

1. Shanghai Innovation Institute
2. Xiamen University
