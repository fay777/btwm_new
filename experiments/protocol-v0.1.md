# BTWM Experiment Protocol v0.1

This protocol defines the minimum reproducible setup for comparing BTWM against
DreamerV3 and, when available, LeWM. It is written to lock the experiment
matrix, reporting rules, and smoke-test gate before server access is available.

## 1. Research Question

Primary question:

Does BTWM improve online world-model RL performance over DreamerV3, and how
does it compare qualitatively against LeWM-style latent world models on tasks
that stress continuous control and long-horizon pixel reasoning?

Primary claims to test:

1. BTWM improves return over a parameter-matched DreamerV3 baseline.
2. The gain is not explained by extra parameter count or extra forward compute.
3. The gain transfers beyond DMC to a harder pixel task family.

## 2. Benchmark Plan

### 2.1 Main benchmark families

1. `DMC`
   - Purpose: continuous control benchmark.
   - Initial tasks:
     - `walker_walk`
     - `cheetah_run`
     - `reacher_easy`
     - `finger_spin`

2. `Crafter`
   - Purpose: long-horizon, sparse-reward, pixel-based environment.
   - Initial task:
     - `crafter_reward`

3. `LeWM comparison set`
   - Purpose: compare against LeWM on tasks where LeWM already has a natural
     training/evaluation story.
   - Initial candidate:
     - `reacher`
   - Current launcher:
     - `code/run_lewm_experiment.sh` with `DOMAIN=dmc TASK=reacher`
   - Note:
     - This comparison is not yet protocol-complete in this repo because LeWM
       uses a separate offline-data and planning pipeline. Until alignment is
       finalized, report LeWM results in a separate table labeled
       "cross-framework comparison".

### 2.2 Priority order

1. `DMC walker_walk`
2. `DMC full 4-task suite`
3. `Crafter reward`
4. `LeWM comparison set`

## 3. Compared Methods

### 3.1 Online RL comparison

1. `DreamerV3 matched baseline`
   - Uses the same inverse head instantiation as BTWM.
   - Auxiliary weights are zeroed.
   - Purpose: parameter-count and forward-compute matched control baseline.

2. `BTWM v2`
   - Default formal variant in this repo.
   - Uses inverse loss on shared features.

3. `BTWM v3` (optional ablation / extension)
   - Confidence-gated and policy-weighted version.
   - Not the default mainline result unless explicitly promoted later.

### 3.2 Cross-framework comparison

1. `BTWM`
2. `DreamerV3 matched baseline`
3. `LeWM`

Cross-framework comparisons must be labeled separately because:

1. BTWM/DreamerV3 are online RL agents.
2. LeWM is an offline world-model + planner pipeline.
3. Data budget and evaluation protocol require explicit alignment.

## 4. Fixed Training Configuration

Unless a table explicitly says otherwise:

### 4.1 DMC

1. Config: `dmc_vision`
2. Steps: `1,000,000`
3. Replay size: `300,000`
4. Batch size: `8`
5. Batch length: `32`
6. Report length: `32`
7. Seed set for formal runs: `0, 1, 2, 3, 4`
8. Variant default: `BTWM_VARIANT=v2`

### 4.2 Crafter

1. Config: `crafter`
2. Steps: `1,000,000`
3. Replay size: `300,000`
4. Seed set for formal runs: `0, 1, 2, 3, 4`
5. Variant default: `BTWM_VARIANT=v2`

### 4.3 Matched baseline rule

For every BTWM run, the paired DreamerV3 baseline must be started with:

1. `--agent.use_btwm True`
2. `--agent.inv_loss_weight 0.0`
3. `--agent.inv_head_loss_weight 0.0`
4. `--agent.inv_policy_weight 0.0`

This preserves architecture and forward compute while removing the BTWM
optimization effect.

## 5. Metrics and Reporting

### 5.1 Primary metrics

1. `Final return`
   - Mean of the last `10` evaluation episodes when available.

2. `Learning curve`
   - Return vs environment steps.

3. `Seed aggregate`
   - Mean and standard deviation across seeds.

### 5.2 Secondary metrics

1. `AUC` over the learning curve.
2. `Best-so-far return`.
3. `Time-to-threshold` if a reasonable threshold exists per task.

### 5.3 BTWM diagnostics

1. `train/inv/ce`
2. `train/acc/inv`
3. `train/inv/objective`
4. `train/inv/valid_frac`
5. `train/loss/inv`

For v3-only runs, also track:

1. `train/inv/confidence`
2. `train/inv/effective_rep_weight`
3. `train/inv/policy_weight_mean`

### 5.4 Plotting rules

1. Every main figure must show steps on the x-axis.
2. Every result table must state seeds and total steps.
3. Cross-framework results must be separated from same-framework comparisons.

## 6. Statistical Plan

For formal tables:

1. Report mean over seeds.
2. Report standard deviation over seeds.
3. If enough tasks are included, additionally report IQM across tasks.
4. Bootstrap confidence intervals are preferred for aggregate benchmark plots.

## 7. Ablation Plan

Minimum ablations:

1. `Matched DreamerV3 baseline`
2. `BTWM v2`
3. `BTWM v3` (if retained)
4. `inv_loss_weight` sweep:
   - `0.0`
   - `0.02`
   - `0.05`
   - `0.10`
5. `inv_feature_source` sweep:
   - `tokens`
   - `stoch`

Optional:

1. Fewer/more inverse bins.
2. Head-only vs shared-feature auxiliary updates.

## 8. Fairness Rules

1. Compare BTWM against a parameter-matched DreamerV3 baseline for all main
   claims.
2. Do not compare resumed runs against fresh runs.
3. Do not mix old pre-fix checkpoints with new corrected inverse-loss runs.
4. Keep run directories unique per task/model/seed/variant.
5. LeWM comparisons must explicitly document:
   - data source,
   - planner budget,
   - observation modality,
   - evaluation horizon.
6. LeWM training runs should be launched through `code/run_lewm_experiment.sh`
   so that logs and run naming stay aligned with the rest of the repository.

## 9. Smoke-Test Gate

Before launching any multi-day server job, the following must pass locally:

1. Unit tests:
   - `python -m unittest code.test_btwm_head`
2. Syntax checks:
   - `python -m py_compile code/btwm_head.py`
   - `python -m py_compile dreamerv3/dreamerv3/agent.py`
   - `python -m py_compile dreamerv3/dreamerv3/rssm.py`
3. Command generation dry-run:
   - `run_experiment.sh` for `btwm/dmc`
   - `run_experiment.sh` for `dreamerv3/crafter`
   - `run_lewm_experiment.sh` for `lewm/dmc/reacher`
4. Optional CPU debug train:
   - one short `debug` config run on `dummy_disc`

No long server run should start until the smoke-test gate is green.

## 10. Immediate Next Steps

1. Finalize the LeWM comparison protocol.
2. Add result-table templates and plotting scripts.
3. Run local smoke tests.
4. Once server access is available, start with:
   - `DreamerV3 vs BTWM` on `dmc_walker_walk`
   - then scale to the full DMC suite
   - then add Crafter
