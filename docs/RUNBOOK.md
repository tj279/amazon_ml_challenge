# RUNBOOK

**Authoritative step-by-step commands.** Use this on every new machine
or new Claude session. Read `docs/STATUS.md` first — it lists current
state, hard-fail bugs to avoid, and the v3 blocker design rationale.

## Table of contents

1. First-time setup (one time per machine)
2. Each new Claude session — start with these reads
3. Phase C v3 — blocking (S2/S3 separately; run them on different machines if you want)
4. Phase C2 — combine S2+S3 and validate recall
5. Phase E — LightGBM training (small-Optuna → scoring → final pipeline; 1.5-2 h on 8 vCPU / 32 GiB)
6. Phase F — inference (test blocker cached once)
7. Phase G + H — graph refinement, packaging

---

## 1. First-time setup (one time per machine)

**a. Python packages** (only `pip install` step needed; libpostal is already
installed on the AWS box the user has been using):

```bash
python3 -m pip install --user polars pandas numpy scikit-learn \
    lightgbm optuna faiss-cpu networkx rapidfuzz datasketch pyarrow \
    unidecode regex indic-transliteration joblib tqdm
```

**b. Smoke check** before any blocking/training run:

```bash
python -c "import polars, numpy, lightgbm, optuna, rapidfuzz; print('OK')"
ls code/business_entity_resolution/artifacts/s{1,2,3}_norm_train.parquet
# All three files must exist. If missing, re-run Phase B (libpostal normalize).
```

---

## 2. Each new Claude session — mandatory first reads

Before doing **anything**, the next Claude session MUST read these three files in this order:

1. `CLAUDE.md` (at the project root) — what we're building, current phase status, hardware realities.
2. `docs/STATUS.md` — per-phase detailed state, **v3 blocker design spec, hard-fail bugs, expected recalls**.
3. `docs/RUNBOOK.md` (this file) — concrete commands.

Then verify what's already done:

```bash
ls code/business_entity_resolution/artifacts/
# Expect: s{1,2,3}_norm_train.parquet (Phase B output)
# Expect: block_S2_features.parquet OR NOT (depending on whether v3 S2 has run)
# If block_features parquets present for both S2 and S3 → Phase C done, go to Phase E.
# If only one direction done → run the missing one.
```

Then check disk and RAM:

```bash
df -h code/business_entity_resolution/artifacts/  # need ~30 GB free
free -h                                          # need 8 GB minimum for blocking
```

---

## 3. Phase C v3 — blocking

**This script already exists at `code/business_entity_resolution/scripts/block_features.py`**.
If it doesn't (e.g., fresh repo clone), copy from the version committed to git.

### 3a. S2-only run (on this machine)

```bash
cd code/business_entity_resolution

# Smoke test on 5% slice, max 3 chunks. Validates code path & RAM.
python scripts/block_features.py \
    --candidate-source S2 --dry-run --max-chunks 3
# Expect: ~1-2 min wall-clock, 3 chunk lines printed in <30 s each, NO final parquet written.
# Clean up smoke chunks before the real run:
rm -rf artifacts/_chunks

# Full S2 run
python scripts/block_features.py \
    --candidate-source S2
# Writes: artifacts/block_S2_features.parquet (~50M rows × 40 cols, ~3 GB on disk).
# Wall-clock on AWS t3.medium (2 vCPU / 8 GiB):    ~50-80 min
# Wall-clock on 8 vCPU / 32 GiB (this user's box): ~30-50 min
```

### 3b. S3 run (on the SAME machine after S2 finishes, OR a different machine)

```bash
python scripts/block_features.py \
    --candidate-source S3
# Writes: artifacts/block_S3_features.parquet (similar size).
```

### 3c. Test blocker caching (Phase F prep, run ONLY when needed)

```bash
# Test normalization MUST be re-run first because Phase B only ran on train.
# Set TRAIN_ONLY=False at the top of src/config.py (one-time change).
# Then run normalize_submit_files() in src/normalize.py to produce
# artifacts/s{1,2,3}_norm_test.parquet.
python -m src.normalize  # with TRAIN_ONLY=False — one-time cost ~10 min on AWS.

# Block on test data with the SAME v3 design.
python scripts/block_features.py \
    --candidate-source S2 --suffix S2_TEST
python scripts/block_features.py \
    --candidate-source S3 --suffix S3_TEST
# Writes artifacts/block_S{2,3}_TEST_features.parquet (each ~1.5-2 GB).
# Wall-clock per direction on t3.medium: ~30 min; on 8 vCPU / 32 GB: ~15-20 min.
# IMPORTANT: do this ONCE and cache. Subsequent model iterations just
# load these parquets in `predict.py` — no re-blocking.
```

### 3d. CLI flags reference

| Flag | Default | Purpose |
| --- | --- | --- |
| `--candidate-source` | (required) | `S2`, `S3`, `S2_TEST`, `S3_TEST` (any custom suffix is fine) |
| `--top-k` | `50` | Final per-S1 cap (HARD 50 per STATUS.md; do NOT lower without recall re-check) |
| `--top-k-index` | `50` | Per-index candidates per S1 (before union + quality filter) |
| `--chunk-size` | `10000` | S1 rows per chunk (raise to lower RAM; lower to fit tighter boxes) |
| `--bucket-cap` | `500` | Max cand ids per structural-key bucket |
| `--token-cap` | `500` | Max cand ids per word-token bucket (Index 2) |
| `--sn-window` | `50` | Sorted-token neighborhood window ±W (101 candidates per S1) |
| `--max-chunks` | `None` | Run only the first N chunks (for smoke testing) |
| `--dry-run` | `False` | Process a 5% slice and skip final concat |
| `--suffix` | (auto) | Output suffix; `--suffix S2_TEST` writes `block_S2_TEST_features.parquet` |

Removed in v3 (kept here for reference only):
| `--trigram-cap` | (n/a) | v2 had char-trigrams; v3 uses word-tokens instead |
| `--top-k-tfidf` | (n/a) | v2's separate per-probe pre-cap; v3 uses a single `--top-k-index` |

### 3e. RAM diagnosis

If you OOM:

- Lower `--chunk-size` to 5000 → halves working-set RAM, ~same speed (probes are polars-vectorized).
- Lower `--bucket-cap` to 200 → halves structural-index RAM (~700 MB → ~350 MB).
- Lower `--token-cap` to 200 → halves token-index RAM (~900 MB → ~450 MB).
- Lower `--sn-window` to 25 → halves sorted-neighborhood candidates; minor gain (~10 s less per chunk).

The v3 polars-vectorized implementation uses **~2.5 GB peak** on the 8 GiB t3.medium box (indexes are polars DFs, not Python dicts). A 5 GB box is theoretically sufficient; 8 GB is comfortable.

---

## 4. Phase C2 — combine S2+S3 + validate recall

### 4a. Combine S2 + S3 blocked parquets

```bash
cd code/business_entity_resolution
python scripts/combine_block_features.py
# Writes: artifacts/block_features.parquet (~110 M pairs × 36 cols, ~3 GB).
```

### 4b. Validate recall (mandatory sanity check)

```bash
python scripts/validate_block_recall.py \
    --blocks artifacts/block_S2_features.parquet artifacts/block_S3_features.parquet
# (or pass --blocks artifacts/block_features.parquet after combine)
# Expect: overall recall ≥ 0.85. Exit code 0 if recall target met, else 2.
```

If recall < 0.80, do NOT proceed to Phase E. Debug:

1. Print per-bucket recall (`U.S.|S2`, `U.S.|S3`, `India|S2`, `India|S3`).
2. If `U.S.|S2` recall is high but `India|S2` low → sorted-token neighborhood may need a bigger window for transliterated names; consider raising `--sn-window` to 100.
3. If both axes are uniformly low → quality-tier filter is too strict; lower floor B from `n_tokens_shared ≥ 2` to `≥ 1`.
4. Inspect `artifacts/match_insights.md` and `artifacts/norm_similarity_insights.md` for the data-driven thresholds.

---

## 5. Phase E — LightGBM training (small-Optuna → scoring → final pipeline)

**Hardware target: 8 vCPU / 32 GiB RAM / 50 GB SSD.** Do **not** try to train on AWS t3.medium (8 GiB) — it will OOM. Total wall-clock on this box: **~1.5-2 h**.

The pipeline is **7 sequential steps**. Each step is resumable (checks for `.done` flag or output artifact) and safe to run detached (`setsid nohup bash -c "..." > /tmp/<step>.log 2>&1 &`).

> **F_0.5 is set-level.** Do **not** use `macro_f05` as a LightGBM early-stopping metric — use `binary_logloss`. Threshold sweep happens **inside each Optuna trial** and again at the post-train stage.

### 5a. Prepare training data (per direction)

```bash
cd code/business_entity_resolution

# For S2
python scripts/prepare_training_data.py --candidate-source S2
# For S3
python scripts/prepare_training_data.py --candidate-source S3
```

- Reads: `artifacts/block_S{2,3}_features.parquet` + `dataset/train/train_ground_truth.tsv`.
- Joins on `(source1_entity_id, candidate_entity_id)`, fills `is_match = 0` for unmatched.
- Adds 6 cheap interaction features (`struct_x_token`, `many_shared_tokens`, `total_signals`, `name_combined`, `addr_combined`, `any_name_missing`).
- Group-aware 90/10 split by `source1_entity_id` (uses `random.Random(42).shuffle` — identical to `validate_block_recall.py`).
- Writes: `artifacts/training_data_S{2,3}.parquet` (~93 M rows × 36 cols, ~3 GB) + `artifacts/val_s1_ids.json`.
- Wall-clock: ~5-10 min/direction.

### 5b. Optuna hyperparameter search (on S2 only — S3 inherits)

```bash
python scripts/optuna_search.py --candidate-source S2 --n-trials 30 --subset-rows 2000000
```

- Subsets to **2 M stratified rows** of S2 training data (15 K positives, 30 s/trial on 8 vCPU; 30 trials ≈ 15 min).
- Sampler: `TPESampler(multivariate=True, seed=42)`. Pruner: `HyperbandPruner(min_resource=20, reduction_factor=3)`.
- Per-trial eval: train → predict val slice → per-(country, source) threshold sweep (0.05-0.95 step 0.05) → return `macro_f05` at best combination.
- Atomic parquet append of `(trial_number, params, value, status, duration_s)` to `artifacts/optuna_trials.parquet` after every trial.
- `--resume` rehydrates via `study.add_trial()` from the parquet.
- Writes: `artifacts/best_params_S2.json`, `artifacts/optuna_trials.parquet`.

### 5c. Train the full-dataset scorer

```bash
python scripts/train_scorer.py --candidate-source S2
```

- Reads `best_params_S2.json`. Trains ONE LightGBM on **10 M stratified S2 rows** (all ~3.7 M positives + 6.3 M random negatives) at best params, `n_estimators=500`.
- Predicts probabilities for the **entire 93 M** S2 training data.
- Writes: `artifacts/lgbm_scorer_S2.txt` (intermediate, can be deleted after 5d), `artifacts/scores_S2.parquet` (one row per pair: `(source1_entity_id, candidate_entity_id, score_p)`).
- Wall-clock: ~10 min train + 5 min predict = 15 min.

### 5d. Drop easy negatives + label noise

```bash
python scripts/sample_training.py --candidate-source S2
```

- Reads `training_data_S2.parquet` + `scores_S2.parquet`.
- Drop: rows where `score_p < 0.05` AND `is_match == 0` (easy negatives).
- Drop: rows where `score_p > 0.95` AND `is_match == 0` (potential GT label noise — false positives).
- Keep: ALL positives (regardless of score) + ALL `score_p ∈ [0.05, 0.95]` rows.
- Writes: `artifacts/sampled_training_S2.parquet` + `artifacts/sampling_meta_S2.json` (`n_pos`, `n_neg_before`, `n_neg_after`, `n_dropped_easy`, `n_dropped_noise`, `pos_neg_ratio_after`).
- Expected post-sample ratio: 1:3 to 1:8.

### 5e. Train final S2 model (two-seed ensemble + snapshot + calibrator)

```bash
python scripts/train_final_S2.py --candidate-source S2
```

- Reads `sampled_training_S2.parquet` + `best_params_S2.json`.
- Splits cleaned data 99/1 by `source1_entity_id` for calibration holdout.
- Trains TWO LightGBM models at best params:
  - `lgbm_S2_seed42.txt` with `seed=42`
  - `lgbm_S2_seed1234.txt` with `seed=1234`, **different** `feature_fraction_seed` and `bagging_seed`
- Snapshot averaging: LightGBM callbacks save model snapshots at rounds `[0.5n, 0.6n, 0.7n, 0.8n, 0.9n, 1.0n]` of total; predictions averaged across the last 3 snapshots per seed.
- Fits `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` on the 1% calibration slice; saves to `artifacts/calibrator_S2.joblib`.
- `n_estimators` = max(1500, 1.5 × Optuna-best). Early stop on `binary_logloss` of the 1% slice (patience=50).
- Writes: `artifacts/lgbm_S2_seed42.txt`, `artifacts/lgbm_S2_seed1234.txt`, `artifacts/calibrator_S2.joblib`, `artifacts/final_meta_S2.json`.
- Wall-clock: ~45 min (two seeds × ~20 min + snapshot overhead + calibration).

### 5f. Transfer-learn to S3

```bash
python scripts/train_S3.py --candidate-source S3
```

- Repeats 5a, 5c, 5d, 5e for S3 (S2's Optuna-best params are inherited — no S3-specific search).
- S3-specific step: `init_model=lgbm_S2_seed42.txt`, `learning_rate=0.02` (half — gentle continuation), `n_estimators=500` cap. Same for `lgbm_S2_seed1234.txt`.
- Recomputes `scale_pos_weight` on S3 sample (different S3 pos:neg ratio).
- Writes: `artifacts/lgbm_S3_seed42.txt`, `artifacts/lgbm_S3_seed1234.txt`, `artifacts/calibrator_S3.joblib`, `artifacts/final_meta_S3.json`.
- Wall-clock: ~30 min combined.

### 5g. Tune per-(country, source) thresholds

```bash
python scripts/threshold.py --candidate-source S2
python scripts/threshold.py --candidate-source S3
```

- Reads each direction's calibration-val slice (1% holdout from Step 5e).
- For each `(country, source)` bucket, sweeps thresholds on grid 0.05-0.95 step 0.01 (101 points) and picks the threshold maximizing bucket macro F_0.5.
- Optional second pass: per-(country, source, is_singleton) — splits each bucket into singleton/non-singleton thresholds. Marginal gain; do it if main sweep plateaus.
- Verifies overall macro F_0.5 improves over a global 0.5 threshold (target: +0.005–0.02 improvement).
- Writes: `artifacts/per_country_source_threshold_S{2,3}.json`, `artifacts/threshold_meta_S{2,3}.json` (F_0.5 at default 0.5 vs swept).
- Wall-clock: 5 min/direction.

### 5h. End-to-end sanity check

```bash
python scripts/validate_block_recall.py \
    --blocks artifacts/block_S2_features.parquet artifacts/block_S3_features.parquet \
    --models artifacts/lgbm_S2_seed42.txt artifacts/lgbm_S2_seed1234.txt \
             artifacts/lgbm_S3_seed42.txt artifacts/lgbm_S3_seed1234.txt \
    --thresholds artifacts/per_country_source_threshold_S2.json \
                  artifacts/per_country_source_threshold_S3.json
# (validate_block_recall.py will need a small flag-add for --models + --thresholds;
# see its current CLI in scripts/validate_block_recall.py — extend if missing.)
# Expect: overall macro F_0.5 ≥ 0.55 on val S1s (stretch 0.65+).
```

---

## 6. Phase F — inference on test data

**Test blocker cached** (see §3c). Iterate cheaply:

```bash
cd code/business_entity_resolution

# ONE-TIME-SETUP (if not already done):
#   - TRAIN_ONLY=False in src/config.py
#   - artifacts/s{1,2,3}_norm_test.parquet exist
#   - artifacts/block_S{2,3}_TEST_features.parquet exist

# Apply model + threshold + singleton to test data:
python scripts/predict.py
# Reads: block_S{2,3}_TEST_features.parquet + lgbm_classifier.txt + threshold JSON.
# Writes: output/matching_results.tsv + output/candidate_pairs.tsv.
# Wall-clock: 5-10 min per iteration.

# Validate submission format:
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
# Exit 0 = submission format OK.
```

**Key speedup trick**: after the first test-blocker run, every model
iteration is **5-10 min** — load parquets, LightGBM predict, threshold,
format. Do NOT re-block on test data unless you change the blocker.

---

## 7. Phase G + H — graph refinement and packaging

(Not yet written; detail to come after Phase F is verified.)

```bash
# Phase G (post Phase F):
python scripts/graph_refine.py
# Closes triangles where min(p_AB, p_BC) ≥ 0.85 with damping 0.9.

# Phase H (final submission):
python scripts/predict.py  # re-run with refined predictions
python utils/validate_submission.py ... (same as Phase F)
zip -r submission.zip output/ code/business_entity_resolution/ Documentation_template.md
```

---

## 8. Common pitfalls (recap from STATUS.md)

- **Polars `.str.split(...)`** on a nullable String column: wrap with `fill_null("")` first.
- **`str.len_chars().abs().cast(Int32)` overflows**: cast to Int32 BEFORE abs.
- **Don't `.select(["source1_entity_id"])`** before adding blocking key columns (loses addr_city etc.).
- **Country hard filter is mandatory**: drop cross-country candidates at probe time.
- **Quality-tier filter is OR, not AND**: a candidate is kept if it passes at least one of the four floors.
- **Structural contribution is capped at 3** in composite score (`min(n_struct_keys, 3) / 3`), not 8.
- **AWS t3.medium is enough for blocking** but NOT for training. Move to a ≥32 GiB box for Phase E.
- **`--trigram-cap` was a v2 flag**: v3 dropped char-trigrams in favor of word-tokens (`--token-cap`). Don't pass `--trigram-cap`; it no longer exists.
- **Clean `artifacts/_chunks/` between runs**: the script auto-cleans at start (real runs only, not `--dry-run`), so a smoke test followed by a real run will discard smoke chunk files automatically. But if you ever interrupt with `Ctrl-C`, do `rm -rf artifacts/_chunks/*` before restarting.
- **Probe results must be unionable**: the three probes return DFs with the same 3 cols (`source1_entity_id, candidate_entity_id, <metric>`); the script `full`-joins them with `coalesce=True`. If you ever modify a probe, keep this 3-col schema invariant or the union breaks.

---

## 9. Memory + time budget (quick reference)

| Stage | Min CPU | Min RAM | Disk | Wall-clock (this box: 8 vCPU / 32 GiB) | Wall-clock (AWS t3.medium: 2 vCPU / 8 GiB) |
| --- | --- | --- | --- | --- | --- |
| Phase C v3 (per direction, vectorized) | 2 vCPU | **8 GiB** | 30 GB | **30-50 min** | 50-80 min |
| Phase C2 (combine + validate) | 2 vCPU | 8 GiB | 5 GB | <5 min | <5 min |
| Phase E (LightGBM training) | 8 cores | **32 GiB** | 50 GB | **1.5–2 h** (small-Optuna → scoring → final pipeline; this box fits) | OOMs on t3.medium |
| Phase F (one-time test block) | 2 vCPU | 8 GiB | +3 GB | ~25 min/direction | ~50 min/direction |
| Phase F (model iteration) | 2 vCPU | 8 GiB | 0 | 5-10 min | 5-10 min |
| Phase G | 4 cores | 16 GiB | +1 GB | ~15 min | ~30 min |
| Phase H (packaging) | 2 vCPU | 8 GiB | +1 GB | <5 min | <5 min |

Speedup notes:
- v3 vectorized per-chunk time: ~10-15 s (8 vCPU) vs ~60 s (Python-loop original).
  The vectorization (polars `DataFrame` indexes + SIMD joins) is the dominant
  reason the times dropped on the same hardware.
- The 8 vCPU / 32 GB box does *not* speed up per-chunk time proportionally to
  CPU count beyond ~2-4 cores (the polars ops already use all cores, but the
  rapidfuzz Python loop in `compute_fuzzy_features` is still single-threaded).
  Most of the gain on this box over t3.medium comes from RAM headroom (32 vs 8 GB)
  not from the extra vCPUs.
