"""Centralized configuration for the Business Entity Resolution pipeline.

Paths, hyper-parameters, country lists, and feature flags live here so every
module reads from a single source of truth. Paths resolve relative to this
file (`Path(__file__).resolve().parents[N]`) so the config is portable across
machines — the previous hardcoded `/home/rujul/...` path did not exist on
the user's 8 vCPU / 32 GB box.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Development policy
# ---------------------------------------------------------------------------

# All development (EDA, normalization, blocking, feature engineering, training,
# validation, threshold tuning, error analysis) MUST operate on training files
# only. The test files in dataset/test/ are not to be read, inspected, or
# processed locally. The test set is touched ONCE, at the very end, by the
# inference script `predict.py` (or pipeline.py --mode submit) to generate
# output TSVs for the leaderboard submission.
#
# Set TRAIN_ONLY = False ONLY when running the final inference / submission
# generation step. Never set it to False during development, validation, or
# model selection — that would defeat the purpose of having a held-out test set.
TRAIN_ONLY: bool = True


# ---------------------------------------------------------------------------
# Paths (resolved relative to this file for portability)
# ---------------------------------------------------------------------------

_THIS_FILE: Path = Path(__file__).resolve()
# config.py lives at .../code/business_entity_resolution/src/config.py
#   parents[0] → .../src
#   parents[1] → .../business_entity_resolution   (PKG_ROOT)
#   parents[2] → .../code
#   parents[3] → .../amazon_ml_challenge         (PROJECT_ROOT)
PKG_ROOT: Path = _THIS_FILE.parents[1]
PROJECT_ROOT: Path = _THIS_FILE.parents[3]
DATA_ROOT: Path = PROJECT_ROOT / "dataset"
DATASET_ROOT: Path = DATA_ROOT

# Final output goes to BOTH the project root output/ (for the official zip)
# and code/business_entity_resolution/output/ (so the code package is
# self-contained per the submission rules).
OUTPUT_ROOT: Path = PROJECT_ROOT / "output"
PKG_OUTPUT_ROOT: Path = PKG_ROOT / "output"

# Code package itself
SRC_ROOT: Path = PKG_ROOT / "src"
ARTIFACTS_ROOT: Path = PKG_ROOT / "artifacts"
NOTEBOOKS_ROOT: Path = PKG_ROOT / "notebooks"
SCRIPTS_ROOT: Path = PKG_ROOT / "scripts"

# Source files
TRAIN_DIR: Path = DATASET_ROOT / "train"
TEST_DIR: Path = DATASET_ROOT / "test"

TRAIN_SOURCE1: Path = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2: Path = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3: Path = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT: Path = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1: Path = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2: Path = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3: Path = TEST_DIR / "test_source3.tsv"

# Helper scripts
VALIDATOR: Path = PROJECT_ROOT / "utils" / "validate_submission.py"
DOC_TEMPLATE: Path = PROJECT_ROOT / "Documentation_template.md"

# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------

# Training countries seen in ground truth
TRAIN_COUNTRIES: List[str] = ["US", "India", "France"]  # open-set
# Test will additionally include France — never hard-code to {US, India}
# Treat country as an open-set string label.

# ---------------------------------------------------------------------------
# Validation / train/val split
# ---------------------------------------------------------------------------

VAL_FRAC: float = 0.10  # fraction of S1 IDs held out for validation
SEED: int = 42

# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------

BLOCKING_TOPK_PER_S1: int = 100  # cap on candidates per S1
BLOCKING_TARGET_RECALL: float = 0.92

# TF-IDF parameters (per country)
TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 3)
TFIDF_MIN_DF: int = 2
TFIDF_MAX_DF: float = 0.95
TFIDF_MAX_FEATURES: int = 1 << 21

# MinHash LSH
MINHASH_THRESHOLD: float = 0.7
MINHASH_PERM: int = 128
MINHASH_SHINGLE_K: int = 3  # char n-gram size

# ---------------------------------------------------------------------------
# Model hyperparameters (canonical — see STATUS.md §Phase E)
# ---------------------------------------------------------------------------

@dataclass
class LightGBMParams:
    """Canonical LightGBM parameters. STATUS.md §Phase E is authoritative.

    These are the values Optuna starts from; OptunaSearchSpace below defines
    the ranges Optuna sweeps. `scale_pos_weight = -1.0` is a sentinel meaning
    "compute dynamically from the sampled training data as n_neg/n_pos".
    See STATUS.md hard-fail bug #10.
    """
    objective: str = "binary"
    metric: List[str] = field(default_factory=lambda: ["binary_logloss", "auc"])
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 1000
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    max_bin: int = 255
    bin_construct_sample_cnt: int = 200_000
    scale_pos_weight: float = -1.0   # -1 → compute dynamically (n_neg/n_pos after sampling)
    verbose: int = -1
    n_jobs: int = 8                  # physical cores on this box (was 18 — stale)
    seed: int = SEED


@dataclass
class OptunaSearchSpace:
    """Search ranges for Optuna in scripts/optuna_search.py.

    Values are (low, high) tuples. Log-scale parameters are marked.
    """
    num_leaves: Tuple[int, int] = (31, 255)
    min_data_in_leaf: Tuple[int, int] = (100, 5000)
    learning_rate: Tuple[float, float] = (0.01, 0.10)        # log-scale
    feature_fraction: Tuple[float, float] = (0.6, 1.0)
    bagging_fraction: Tuple[float, float] = (0.6, 1.0)
    bagging_freq: Tuple[int, int] = (1, 10)
    lambda_l1: Tuple[float, float] = (0.0, 5.0)
    lambda_l2: Tuple[float, float] = (0.0, 5.0)
    max_bin: Tuple[int, int] = (127, 511)


# Optuna search runtime config
OPTUNA_N_TRIALS: int = 30
OPTUNA_TIMEOUT_S: int = 7200          # 2 h hard cap
OPTUNA_SUBSET_ROWS: int = 2_000_000  # stratified subset size for Optuna (see RUNBOOK §5b)
OPTUNA_DIRECTION: str = "maximize"    # we maximize macro_f05

# Two-seed ensemble for final model
ENSEMBLE_SEEDS: List[int] = [42, 1234]

# S3 transfer-learn hyperparameters (continues boosting from S2 model)
S3_TRANSFER_LR: float = 0.02          # half the canonical rate — gentle continuation
S3_TRANSFER_N_ESTIMATORS: int = 500  # cap; transfer inherits most trees from S2

# ---------------------------------------------------------------------------
# Per-country, per-source default thresholds (starting points — sweep on val)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: Dict[Tuple[str, str], float] = {
    ("US", "S2"): 0.55,
    ("US", "S3"): 0.55,
    ("India", "S2"): 0.50,
    ("India", "S3"): 0.55,
    ("France", "S2"): 0.65,
    ("France", "S3"): 0.65,
}

# Threshold sweep config (scripts/threshold.py)
THRESHOLD_SWEEP_LOW: float = 0.05
THRESHOLD_SWEEP_HIGH: float = 0.95
THRESHOLD_SWEEP_STEP: float = 0.01
THRESHOLD_SWEEP_PER_SINGLETON: bool = True   # also sweep per-(country, source, is_singleton)

# ---------------------------------------------------------------------------
# Singleton handling — no separate detector
# ---------------------------------------------------------------------------

# The set-level F_0.5 contract (`src/f05.py::f05_single` returns 1.0 if pred
# empty else 0.0) already gives full credit for correctly predicting singletons.
# A separate `singleton_detector.py` would just retrain what the threshold
# sweep encodes. No SINGLETON_PROB_THRESHOLD needed; per-(country, source,
# is_singleton) threshold buckets handle this if needed.

# ---------------------------------------------------------------------------
# Graph refinement
# ---------------------------------------------------------------------------

GRAPH_TRIANGLE_MIN_PROB: float = 0.85
GRAPH_DAMPING: float = 0.9
GRAPH_MAX_NEW_EDGE_RATE: float = 0.03  # if closure adds >3% new edges, leakage

# ---------------------------------------------------------------------------
# Cascade (high-precision auto-commit)
# ---------------------------------------------------------------------------

CASCADE_NAME_LEV_THRESHOLD: float = 0.92
CASCADE_NAME_JARO_THRESHOLD: float = 0.96
CASCADE_NAME_TOKEN_SET_THRESHOLD: float = 0.95
CASCADE_NAME_JACCARD_THRESHOLD: float = 0.7

# ---------------------------------------------------------------------------
# Hard-negative mining (2-stage; see STATUS.md §Phase E + hard-fail bug #11)
# ---------------------------------------------------------------------------

# Step 1: train a scorer at Optuna-best params on a stratified sample
SAMPLE_SIZE_SCORER: int = 10_000_000   # 10 M rows — all positives + random negs

# Step 2: drop rows based on scorer probability
SCORE_DROP_THRESHOLD: float = 0.05     # below this = easy negative → drop (if is_match == 0)
SCORE_NOISE_THRESHOLD: float = 0.95    # above this + is_match == 0 = potential label noise → drop

# Post-sample balance expectation
TARGET_POS_NEG_RATIO_MIN: float = 1/8  # 1:8 negs per pos = upper bound
TARGET_POS_NEG_RATIO_MAX: float = 1/3  # 1:3 negs per pos = lower bound

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs(OUTPUT_ROOT, PKG_OUTPUT_ROOT, ARTIFACTS_ROOT, NOTEBOOKS_ROOT, SCRIPTS_ROOT)
