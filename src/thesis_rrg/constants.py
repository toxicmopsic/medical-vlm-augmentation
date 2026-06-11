from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "thesis-radiology-report-gen"
PACKAGE_NAME = "thesis_rrg"

DEFAULT_PROMPT_TEMPLATE = "<start_of_image> {indication} findings:"
DEFAULT_REPORT_SEPARATOR = " impression: "

DEFAULT_SPLITS = ("train", "validate", "test")

RUN_SUBDIRS = (
    "checkpoints/last",
    "checkpoints/best",
    "predictions",
    "metrics",
)

DEFAULT_HYDRA_CONFIG_NAME = "hydra_config.yaml"
DEFAULT_GIT_COMMIT_FILE = "git_commit.txt"
DEFAULT_TRAIN_LOG_FILE = "train.log"

DEFAULT_OUTPUT_ROOTS = {
    "train": "train",
    "eval": "eval",
    "predict": "predict",
    "prepare": "prepare",
}

CACHE_FILES = {
    "prepared": "mimic_prepared_frontal1perstudy_sections_{cache_tag}.parquet",
    "clean": "mimic_prepared_frontal1perstudy_sections_{cache_tag}_clean.parquet",
    "e2a": "mimic_e2a_struct_targets_{cache_tag}.parquet",
}

MIMIC_REQUIRED_COLUMNS = (
    "subject_id",
    "study_id",
    "dicom_id",
    "split",
    "dcm_path",
    "indication",
    "ref_findings",
    "ref_impression",
)
