# Medical VLM Augmentation for Chest X-ray Report Generation

Research code for studying how data augmentation affects chest X-ray report
generation with a medical vision-language model. The project fine-tunes
MedGemma on MIMIC-CXR and compares visual augmentations, structural text
augmentation, LLM-based report paraphrasing, joint image-text augmentation, and
synthetic pretraining with RoentGen-v2 generated radiographs.

The repository is organized for reproducible experiments: all major runs are
configured with Hydra, train/eval artifacts are written to standardized
directories, and the evaluation pipeline reports lexical, semantic, and
clinically oriented metrics.

> Research use only. This code is not intended for clinical decision-making.

## Highlights

- MedGemma-based CXR report generation pipeline.
- LoRA/QLoRA fine-tuning with Hugging Face `Trainer` and optional TRL SFT.
- MIMIC-CXR preprocessing and report section extraction.
- Visual augmentations: geometric, photometric, mixed, and strong variants.
- Text augmentations: structural target transformations and LLM paraphrases.
- Joint image-text augmentation experiments.
- RoentGen-v2 synthetic CXR pretraining experiment.
- Evaluation with ROUGE-L, SacreBLEU, BERTScore, RaTEScore, F1CheXbert,
  RadGraph F1, GREEN, and RadCliQ-v1.
- Single-GPU, DDP, and FSDP launch examples.

## Repository Layout

```text
configs/              Hydra configs for data, models, training, evaluation, and experiments
src/thesis_rrg/       Python package with CLI entrypoints and experiment code
scripts/              Convenience shell wrappers for common prepare/train/eval runs
Makefile              Thin wrappers around the Python CLIs
pyproject.toml        Package metadata and dependencies
README.md             This file
```

Large generated artifacts are intentionally not versioned:

```text
MIMIC_CACHE_DIR/
E2a_CACHE/
E2c_outputs*/
H4_synthetic_roentgen_v2/
train/
eval/
predict/
*.parquet
*.pt
*.ckpt
```

## Data and Model Access

This repository does not redistribute MIMIC-CXR, trained checkpoints, generated
caches, or synthetic images. To reproduce the experiments, prepare the following
outside the repository:

- Authorized access to MIMIC-CXR DICOM images and reports.
- Access to the required Hugging Face model weights, including MedGemma.
- Optional access to RoentGen-v2 for the synthetic CXR experiment.
- Optional metric dependencies for RadGraph/CheXbert-style evaluation.

## Installation

```bash
git clone git@github.com:toxicmopsic/medical-vlm-augmentation.git
cd medical-vlm-augmentation

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For development and optional metric packages:

```bash
pip install -e ".[dev,metrics]"
```

The project expects Python 3.10+ and a CUDA-enabled PyTorch installation for
training and evaluation.

## Common Path Overrides

Most commands below use the same storage locations. Set them once and pass the
Hydra overrides to every command.

```bash
export ROOT_DIR="$PWD"
export MIMIC_ROOT="/path/to/MIMIC-CXR-DICOM"
export CACHE_DIR="$ROOT_DIR/MIMIC_CACHE_DIR"
export E2A_DIR="$ROOT_DIR/E2a_CACHE"
export E2C_OUT_DIR="$ROOT_DIR/E2c_outputs_v1"

COMMON_OVERRIDES=(
  paths.root_dir="$ROOT_DIR"
  data.mimic_root="$MIMIC_ROOT"
  data.cache_dir="$CACHE_DIR"
  data.e2a_dir="$E2A_DIR"
  data.e2c_out_dir="$E2C_OUT_DIR"
)
```

If you do not use Bash arrays, append these overrides manually to each command.

## Preparing Data

### 1. Build the MIMIC-CXR cache

This step parses MIMIC-CXR studies, extracts report sections, selects the image
view policy used by the experiments, and writes cached parquet files.

```bash
python -m thesis_rrg.cli.prepare_data \
  experiment=baseline_mimic \
  data.prepare_mode=baseline_cache \
  "${COMMON_OVERRIDES[@]}"
```

### 2. Build structural text augmentation targets

Structural target augmentation creates alternative report targets while
preserving the clinical content as much as possible.

```bash
python -m thesis_rrg.cli.prepare_data \
  experiment=h2_e2a_structural \
  data.prepare_mode=e2a \
  data.e2a_use_radgraph=false \
  data.e2a_cache_path="$E2A_DIR/mimic_e2a_struct_targets_v2.parquet" \
  "${COMMON_OVERRIDES[@]}"
```

### 3. Generate LLM paraphrase augmentation targets

The paraphrase pipeline can be sharded across GPUs. The example below launches
four shards; in practice, run them in separate terminals or scheduler jobs if
you want parallel generation.

```bash
for SID in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$SID" python -m thesis_rrg.cli.prepare_data \
    experiment=h2_e2c_llm_paraphrase \
    data.prepare_mode=e2c \
    data.e2c_shard_id="$SID" \
    data.e2c_out_dir="$E2C_OUT_DIR" \
    data.e2c_resume=false \
    data.e2c_max_rows=-1 \
    data.e2c_candidates_per_prompt=4 \
    data.e2c_llm_batch_size=16 \
    data.e2c_max_new_tokens=220 \
    data.e2c_min_new_tokens=16 \
    data.e2c_temperature=0.7 \
    data.e2c_top_p=0.95 \
    data.e2c_repetition_penalty=1.10 \
    data.e2c_no_repeat_ngram=4 \
    "${COMMON_OVERRIDES[@]}"
done
```

Merge the selected candidates after all shards finish:

```bash
python -m thesis_rrg.cli.prepare_data \
  experiment=h2_e2c_llm_paraphrase \
  data.prepare_mode=merge_e2c \
  data.e2c_out_dir="$E2C_OUT_DIR" \
  data.e2c_num_shards=4 \
  'data.e2c_merge_shard_ids=[0,1,2,3]' \
  data.e2c_merged_path="$E2C_OUT_DIR/e2c_best_train_merged_4of4_v1.parquet" \
  "${COMMON_OVERRIDES[@]}"
```

### 4. Prepare synthetic RoentGen-v2 data

This is optional and only needed for the synthetic pretraining experiment.

```bash
CUDA_VISIBLE_DEVICES=0 python -m thesis_rrg.cli.prepare_data \
  experiment=h4_synth_pretrain_real_ft \
  data.prepare_mode=h4_all \
  data.h4_device=cuda:0 \
  data.h4_medsiglip_device=cuda:0 \
  data.h4_roentgen_text_encoder_id=stanfordmimi/RoentGen-v2 \
  data.h4_target_accepted_studies=2000 \
  data.h4_checkpoint_every=25 \
  data.e2c_merged_path="$E2C_OUT_DIR/e2c_best_train_merged_4of4_v1.parquet" \
  "${COMMON_OVERRIDES[@]}"
```

## Training

### Single-GPU baseline

```bash
CUDA_VISIBLE_DEVICES=0 python -m thesis_rrg.cli.train \
  experiment=baseline_mimic \
  training=hf_trainer \
  prompts=findings_impression \
  logging=tensorboard \
  training.per_device_train_batch_size=2 \
  training.per_device_eval_batch_size=2 \
  training.gradient_accumulation_steps=8 \
  data.num_workers=8 \
  "${COMMON_OVERRIDES[@]}"
```

### Visual augmentation

```bash
CUDA_VISIBLE_DEVICES=0 python -m thesis_rrg.cli.train \
  experiment=h1_image_aug_photo \
  data.augmentation_name=photo \
  training=hf_trainer \
  prompts=findings_impression \
  logging=tensorboard \
  training.per_device_train_batch_size=2 \
  training.per_device_eval_batch_size=2 \
  training.gradient_accumulation_steps=8 \
  data.num_workers=8 \
  "${COMMON_OVERRIDES[@]}"
```

### Structural text augmentation

```bash
CUDA_VISIBLE_DEVICES=0 python -m thesis_rrg.cli.train \
  experiment=h2_e2a_structural \
  data.prepare_mode=e2a \
  training=hf_trainer \
  prompts=findings_impression \
  logging=tensorboard \
  data.e2a_cache_path="$E2A_DIR/mimic_e2a_struct_targets_v2.parquet" \
  training.per_device_train_batch_size=2 \
  training.per_device_eval_batch_size=2 \
  training.gradient_accumulation_steps=8 \
  data.num_workers=8 \
  "${COMMON_OVERRIDES[@]}"
```

### LLM paraphrase target augmentation

```bash
CUDA_VISIBLE_DEVICES=0 python -m thesis_rrg.cli.train \
  experiment=h2_e2c_llm_paraphrase \
  data.experiment_mode=e2c \
  training=hf_trainer \
  prompts=findings_impression \
  logging=tensorboard \
  data.e2c_apply_p=0.7 \
  data.e2c_merged_path="$E2C_OUT_DIR/e2c_best_train_merged_4of4_v1.parquet" \
  training.per_device_train_batch_size=2 \
  training.per_device_eval_batch_size=2 \
  training.gradient_accumulation_steps=8 \
  data.num_workers=8 \
  "${COMMON_OVERRIDES[@]}"
```

### Joint image-text augmentation

```bash
CUDA_VISIBLE_DEVICES=0 python -m thesis_rrg.cli.train \
  experiment=h3_e3a_safe_joint \
  data.h3_mode=e3a \
  data.experiment_mode=e3a \
  data.augmentation_name=photo \
  training=hf_trainer \
  prompts=findings_impression \
  logging=tensorboard \
  data.e2c_apply_p=0.7 \
  data.e2c_merged_path="$E2C_OUT_DIR/e2c_best_train_merged_4of4_v1.parquet" \
  training.per_device_train_batch_size=2 \
  training.per_device_eval_batch_size=2 \
  training.gradient_accumulation_steps=8 \
  data.num_workers=8 \
  "${COMMON_OVERRIDES[@]}"
```

### Synthetic pretraining and real fine-tuning

First pretrain on synthetic samples:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29505 \
  -m thesis_rrg.cli.train \
  experiment=h4_synth_pretrain_real_ft \
  data.h4_train_stage=synthetic_pretrain \
  data.h4_artifact_name=h4_main_n2000_k2_seed0_sexprompt_sigmoid2p7e-4 \
  training=hf_trainer_ddp_a40 \
  model.attn_implementation=sdpa \
  training.per_device_train_batch_size=2 \
  training.per_device_eval_batch_size=2 \
  training.gradient_accumulation_steps=4 \
  logging=tensorboard \
  data.num_workers=8 \
  "${COMMON_OVERRIDES[@]}"
```

Then fine-tune on the matched real subset:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29506 \
  -m thesis_rrg.cli.train \
  experiment=h4_synth_pretrain_real_ft \
  data.h4_train_stage=real_finetune \
  data.h4_artifact_name=h4_main_n2000_k2_seed0_sexprompt_sigmoid2p7e-4 \
  model.adapter_path="<path-to-synthetic-pretrain-checkpoint>" \
  training=hf_trainer_ddp_a40 \
  model.attn_implementation=sdpa \
  training.per_device_train_batch_size=2 \
  training.per_device_eval_batch_size=2 \
  training.gradient_accumulation_steps=4 \
  logging=tensorboard \
  data.num_workers=8 \
  "${COMMON_OVERRIDES[@]}"
```

## Multi-GPU Training

DDP is the recommended multi-GPU mode when the model fits on one GPU. It is
usually simpler and faster for LoRA/QLoRA experiments.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29500 \
  -m thesis_rrg.cli.train \
  experiment=h1_image_aug_photo \
  data.augmentation_name=photo \
  training=hf_trainer_ddp_a40 \
  model.attn_implementation=sdpa \
  prompts=findings_impression \
  logging=tensorboard \
  training.per_device_train_batch_size=2 \
  training.per_device_eval_batch_size=2 \
  training.gradient_accumulation_steps=4 \
  data.num_workers=8 \
  "${COMMON_OVERRIDES[@]}"
```

The effective global batch size is:

```text
per_device_train_batch_size * num_gpus * gradient_accumulation_steps
```

For example:

```text
single GPU: 2 * 1 * 8 = 16
2-GPU DDP:  2 * 2 * 4 = 16
```

FSDP is available for memory-constrained runs or larger models:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=29501 \
  -m thesis_rrg.cli.train \
  experiment=h2_e2a_structural \
  data.prepare_mode=e2a \
  training=hf_trainer_fsdp_a40 \
  model.attn_implementation=sdpa \
  prompts=findings_impression \
  logging=tensorboard \
  data.e2a_cache_path="$E2A_DIR/mimic_e2a_struct_targets_v2.parquet" \
  "${COMMON_OVERRIDES[@]}"
```

## Evaluation

Evaluate a trained adapter on the fixed test split:

```bash
CUDA_VISIBLE_DEVICES=0 python -m thesis_rrg.cli.eval \
  experiment=baseline_mimic \
  model/generation=guarded \
  model.adapter_path="<path-to-train-run>/checkpoint-6000" \
  data.eval_split=test \
  data.eval_batch_size=4 \
  data.num_workers=8 \
  "${COMMON_OVERRIDES[@]}"
```

For E2C and joint-augmentation runs, pass the merged paraphrase file:

```bash
CUDA_VISIBLE_DEVICES=0 python -m thesis_rrg.cli.eval \
  experiment=h2_e2c_llm_paraphrase \
  data.experiment_mode=e2c \
  model/generation=guarded \
  model.adapter_path="<path-to-train-run>/checkpoint-6500" \
  data.e2c_merged_path="$E2C_OUT_DIR/e2c_best_train_merged_4of4_v1.parquet" \
  data.eval_split=test \
  data.eval_batch_size=4 \
  data.num_workers=8 \
  "${COMMON_OVERRIDES[@]}"
```

For quick smoke evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m thesis_rrg.cli.eval \
  experiment=baseline_mimic \
  model.adapter_path="<path-to-checkpoint>" \
  data.eval_split=test \
  data.max_test=64 \
  data.eval_batch_size=1 \
  data.num_workers=0 \
  "${COMMON_OVERRIDES[@]}"
```

## Outputs

Training and evaluation runs are written under Hydra-managed output
directories. A typical training run contains:

```text
train/
  2026-05-08_14-57-43_h2_e2c_llm_paraphrase/
    hydra_config.yaml
    git_commit.txt
    train.log
    checkpoint-*/
    tensorboard/
```

A typical evaluation run contains:

```text
eval/
  2026-05-08_.../
    hydra_config.yaml
    eval.log
    predictions/
      test_predictions.csv
      test_generation_flags.csv
      test_length_outliers.csv
    metrics/
      test_metrics.json
      test_generation_pathology_summary.json
      dataset_bundle.json
```

TensorBoard:

```bash
tensorboard --logdir train
```

## Results

Final test-set metrics are reported on the full report text as
`median [95% CI]`. All metrics are scaled to points out of 100. Higher is
better for all columns except RadCliQ-v1, where lower is better.

The H4 rows use a separate matched setting with 2,000 real studies and should be
compared within H4, not directly against H1-H3.

### Lexical and semantic metrics

| Experiment | ROUGE-L (up) | SacreBLEU (up) | BERTScore (up) | RaTEScore (up) |
|---|---:|---:|---:|---:|
| Baseline | 29.3 [28.7, 29.9] | 10.2 [9.9, 10.5] | 55.3 [54.8, 55.8] | 60.6 [60.0, 61.2] |
| H1 geo | 29.5 [28.9, 30.1] | 10.3 [10.0, 10.6] | 55.6 [55.1, 56.1] | 61.0 [60.4, 61.6] |
| H1 mix | 29.6 [29.0, 30.2] | 10.3 [10.0, 10.7] | 55.9 [55.4, 56.4] | 61.2 [60.6, 61.8] |
| H1 photo | **29.8 [29.2, 30.4]** | **10.4 [10.1, 10.7]** | 56.0 [55.5, 56.5] | 61.4 [60.8, 62.0] |
| H2 structural text | **29.8 [29.2, 30.4]** | **10.4 [10.1, 10.7]** | 55.7 [55.2, 56.2] | 61.1 [60.5, 61.7] |
| H2 LLM paraphrase | 28.4 [27.8, 29.0] | 9.9 [9.6, 10.2] | 56.1 [55.6, 56.6] | 61.5 [60.9, 62.1] |
| H3 joint photo + paraphrase | 28.8 [28.2, 29.4] | 10.0 [9.7, 10.3] | **56.7 [56.2, 57.2]** | **62.2 [61.6, 62.8]** |
| H4 real-only* | 27.9 [27.3, 28.5] | 8.9 [8.6, 9.2] | 54.0 [53.5, 54.5] | 60.0 [59.4, 60.6] |
| H4 synth oldprompt + real FT* | 28.0 [27.4, 28.6] | 9.0 [8.7, 9.3] | 54.0 [53.5, 54.5] | 59.7 [59.1, 60.3] |
| H4 synth sexprompt + real FT* | **28.2 [27.6, 28.8]** | **9.4 [9.1, 9.7]** | **54.3 [53.8, 54.8]** | **60.1 [59.5, 60.7]** |

### Clinically oriented metrics

| Experiment | F1CheXbert (up) | RadGraph F1 (up) | GREEN (up) | RadCliQ-v1 (down) |
|---|---:|---:|---:|---:|
| Baseline | 52.7 [51.5, 53.9] | 29.5 [28.5, 30.5] | 36.4 [35.5, 37.3] | 78.1 [76.8, 79.4] |
| H1 geo | 53.1 [51.9, 54.3] | 29.8 [28.8, 30.8] | 36.7 [35.8, 37.6] | 77.8 [76.5, 79.1] |
| H1 mix | 53.3 [52.1, 54.5] | 30.0 [29.0, 31.0] | 36.9 [36.0, 37.8] | 76.9 [75.6, 78.2] |
| H1 photo | 53.4 [52.2, 54.6] | 30.2 [29.2, 31.2] | 37.0 [36.1, 37.9] | 77.0 [75.7, 78.3] |
| H2 structural text | 53.5 [52.3, 54.7] | 29.8 [28.8, 30.8] | 36.9 [36.0, 37.8] | **76.4 [75.1, 77.7]** |
| H2 LLM paraphrase | **54.5 [53.3, 55.7]** | **30.3 [29.3, 31.3]** | 37.0 [36.1, 37.9] | 76.9 [75.6, 78.2] |
| H3 joint photo + paraphrase | 54.0 [52.8, 55.2] | 29.8 [28.8, 30.8] | **37.7 [36.8, 38.6]** | 76.8 [75.5, 78.1] |
| H4 real-only* | **50.3 [49.1, 51.5]** | 26.8 [25.9, 27.7] | 34.9 [34.0, 35.8] | **73.5 [72.2, 74.8]** |
| H4 synth oldprompt + real FT* | 49.8 [48.6, 51.0] | 27.2 [26.3, 28.1] | 34.9 [34.0, 35.8] | **73.5 [72.2, 74.8]** |
| H4 synth sexprompt + real FT* | 50.2 [49.0, 51.4] | **27.7 [26.8, 28.6]** | **35.2 [34.3, 36.1]** | 74.8 [73.5, 76.1] |

## Reproducibility Notes

- The main comparison keeps the model, train/validation/test split, prompt
  format, decoding configuration, and evaluation pipeline fixed.
- H1-H3 are compared against the same full MIMIC-CXR training setup.
- H4 uses a separate matched real-only versus synthetic-pretrain setting.
- Each run stores the Hydra config and Git commit hash in its output directory.
- Generated caches and checkpoints are intentionally excluded from Git.

## Command-Line Entrypoints

The package exposes both module and console-script entrypoints:

```bash
python -m thesis_rrg.cli.prepare_data
python -m thesis_rrg.cli.train
python -m thesis_rrg.cli.eval
python -m thesis_rrg.cli.predict
python -m thesis_rrg.cli.summarize_run
```

or:

```bash
thesis-rrg-prepare
thesis-rrg-train
thesis-rrg-eval
thesis-rrg-predict
thesis-rrg-summarize
```

## License

See `LICENSE`.