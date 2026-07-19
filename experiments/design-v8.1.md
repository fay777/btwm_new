# BTWM — Claim-Driven Experiment Plan v8.1

**Date:** 2026-06-09
**Status:** Design — awaiting boss approval
**Principle:** ARIS experiment-plan — every block maps to a specific claim; maximally efficient compute budget

---

## 1. Claims Matrix

### Primary Claims (PC)

| ID | Claim | Type | Paper Location |
|----|-------|------|----------------|
| **PC-1** | BTWM outperforms LeWorldModel at matched compute across MBRL benchmarks | Performance | §5.1, Table 1 |
| **PC-2** | The inverse-action prediction loss alone accounts for the majority of BTWM's gain over LeWorldModel | Mechanism | §5.4, Table 4 |

### Supporting Claims (SC)

| ID | Claim | Type | Paper Location |
|----|-------|------|----------------|
| SC-3 | The gain is not from extra model capacity (compute-matched LeWorldModel shows no comparable gain) | Exclusivity | §5.4 control |
| SC-4 | The gain is not from an arbitrary auxiliary signal (random-label head shows no gain) | Exclusivity | §5.4 control |
| SC-5 | BTWM's bidirectional representations yield measurable mutual information benefits over forward-only models | Analysis | §5.5, Fig 5 |
| SC-6 | BTWM generalizes to continuous control (DMC, Meta-World) with consistent gains | Generalization | §5.2-5.3 |
| SC-7 | Policy masking and shared trunk provide complementary but smaller gains than inverse loss | Ablation | §5.4 |

### Anti-Claims (tested to refute)

| ID | Anti-Claim | Rebuttal Experiment |
|----|-----------|---------------------|
| AC-1 | "BTWM is just LeWorldModel with better HP tuning" | Compute-matched LeWorldModel control (E3.1) |
| AC-2 | "Any auxiliary head gives the same gain" | Random-label auxiliary head control (E3.2) |
| AC-3 | "Inverse loss is redundant with forward loss" | Mutual information analysis (E6) |

---

## 2. Atari Game Selection: 26 → 10

Selected to cover five distinct challenge categories while preserving benchmark representativeness. Selection rationale: pick the 2 most representative games per category where LeWorldModel shows non-trivial room for improvement.

| # | Game | Category | Action Space | Key Challenge | LeWM Score | Why Included |
|---|------|----------|-------------|---------------|------------|-------------|
| 1 | **Montezuma's Revenge** | Exploration-heavy | 18 discrete | Sparse rewards, long horizons, room navigation | 0.20 | Hardest exploration game; largest headroom |
| 2 | **Private Eye** | Exploration-heavy | 18 discrete | Sparse rewards, navigation, object interaction | 0.50 | Medium-difficulty exploration with fine-grained search |
| 3 | **Breakout** | Fine control | 4 discrete | Precise paddle positioning, ball physics | 0.66 | Canonical fine-control benchmark; inverse loss should help distinguish paddle micro-adjustments |
| 4 | **Pong** | Fine control | 6 discrete | Rapid paddle tracking, adversarial dynamics | 0.76 | Simplest fine-control; tests if inverse helps even when forward is easy |
| 5 | **Alien** | Survival/strategy | 18 discrete | Enemy avoidance, resource management | 0.62 | Complex action space + survival pressure |
| 6 | **Amidar** | Survival/strategy | 10 discrete | Territory capture, enemy patterns | 0.38 | Strategic planning; tests temporal credit assignment |
| 7 | **Qbert** | Puzzle/navigation | 6 discrete | Sequential tile-flipping, falling hazards | 0.62 | Requires action chaining to solve puzzles |
| 8 | **Ms Pacman** | Puzzle/navigation | 9 discrete | Maze navigation, ghost avoidance, pellet collection | 0.58 | Requires planning and pathfinding |
| 9 | **Boxing** | Competitive/shooting | 18 discrete | Adversarial opponent, reactive positioning | 0.78 | Adversarial dynamics; inverse should help read opponent cues |
| 10 | **Seaquest** | Competitive/shooting | 18 discrete | Enemy targeting, oxygen management, multi-tasking | 0.54 | Multi-tasking challenge; high LeWM→BTWM Δ in pilot |

**Coverage justification:** These 10 games span a mean LeWM score of 0.56 (headroom-rich), cover all Atari action space sizes (4–18), and represent categories where inverse-action prediction is hypothesized to help most: exploration (disambiguating state transitions), fine control (micro-action discrimination), and multi-tasking (action-to-outcome attribution).

---

## 3. Experiment Blocks

### Block 1 — Atari 100k Main Results [P0]

**Claim:** PC-1 (BTWM > LeWorldModel at matched compute)

**Description:** Train and evaluate BTWM vs. LeWorldModel, DreamerV3, STORM on the 10-game Atari 100k subset.

| Parameter | Value |
|-----------|-------|
| Games | 10 (see §2) |
| Models | BTWM, LeWorldModel, DreamerV3, STORM |
| Seeds per game × model | 5 |
| Environment steps per game | 100,000 |
| Total runs | 10 × 4 × 5 = **200 runs** |
| Primary metric | IQM human-normalized score over 10 games |
| Secondary metrics | Mean, Median, #Superhuman games, per-game Δ vs LeWorldModel |
| Success criterion | IQM(BTWM) > IQM(LeWorldModel) with p < 0.05 (stratified bootstrap) |
| Wall-time estimate | ~2.5 GPU-days (A100, 4 parallel runs) |

**Output:** Table 1 (aggregate) + Supplementary Table S1 (per-game with 95% CIs).

---

### Block 2 — Ablation Study [P0]

**Claim:** PC-2 (inverse loss accounts for majority of gain) + SC-7

**Description:** Component ablation isolating inverse loss, policy masking, shared trunk, and return relabeling.

| Variant | Inverse Loss | Policy Masking | Shared Trunk | Return Relabeling | Hypothesis |
|---------|:---:|:---:|:---:|:---:|------------|
| Full BTWM | ✓ | ✓ | ✓ | ✓ | Upper bound |
| − Inverse Loss | ✗ | ✓ | ✓ | ✓ | Largest drop predicted |
| − Policy Masking | ✓ | ✗ | ✓ | ✓ | Small drop (complementary) |
| − Shared Trunk | ✓ | ✓ | ✗ | ✓ | Moderate drop (separate encoders) |
| − Return Relabeling | ✓ | ✓ | ✓ | ✗ | Moderate drop |
| Forward-Only (LeWM) | ✗ | ✗ | ✓ | ✓ | Lower bound |

| Parameter | Value |
|-----------|-------|
| Games | 10 |
| Variants | 6 |
| Seeds per variant × game | 5 |
| Environment steps per game | 100,000 |
| Total runs | 10 × 6 × 5 = **300 runs** |
| Metric | IQM human-normalized score |
| Success criterion | IQM drop from −Inverse Loss > IQM drop from any single other ablation, Δ > 0.10 IQM |

**Output:** Table 4 (IQM per variant + Δ column) + Fig 5 (bar chart with 95% CIs).

---

### Block 3 — Control Experiments [P0]

**Claim:** SC-3 (not extra capacity) + SC-4 (not arbitrary signal)

**Description:** Two critical control experiments to establish exclusivity of the inverse-action mechanism.

**E3.1 — Compute-Matched LeWorldModel Control**
| Parameter | Value |
|-----------|-------|
| Variant | LeWorldModel with 2 extra transformer layers (15.1M params, matching BTWM's 15.1M) |
| Games | 10 |
| Seeds | 5 |
| Total runs | 10 × 5 = 50 runs |
| Hypothesis | Adding parameters without inverse loss does not close the gap |
| Success criterion | IQM(compute-matched LeWM) ≤ IQM(LeWM) + 0.03 |

**E3.2 — Random-Label Auxiliary Head Control**
| Parameter | Value |
|-----------|-------|
| Variant | BTWM with inverse head predicting randomly permuted action labels |
| Games | 10 |
| Seeds | 5 |
| Total runs | 10 × 5 = 50 runs |
| Hypothesis | An arbitrary auxiliary signal provides no gain |
| Success criterion | IQM(random-head) ≈ IQM(LeWorldModel), significantly below IQM(BTWM) |

**Combined:** 100 runs. Output: Table 4 footnote + inline text in §5.4.

---

### Block 4 — DMC Generalization [P1]

**Claim:** SC-6 (generalizes to continuous control)

**Description:** Evaluate BTWM on DeepMind Control Suite with continuous action discretization.

| Parameter | Value |
|-----------|-------|
| Tasks | 9 (walker-walk, cheetah-run, reacher-easy, manipulator-bring-ball, finger-spin, cartpole-swingup, ball-in-cup-catch, hopper-hop, acrobot-swingup) |
| Models | BTWM, LeWorldModel, DreamerV3 |
| Seeds per task × model | 5 |
| Environment steps | 500,000 |
| Total runs | 9 × 3 × 5 = **135 runs** |
| Metric | Episode return (mean ± std) |
| Success criterion | BTWM > LeWorldModel on ≥ 7/9 tasks; aggregate mean return gain > 4% |

**Output:** Table 2 (4 representative tasks in main paper, full 9-task table in supplementary).

---

### Block 5 — Meta-World Generalization [P1]

**Claim:** SC-6 (generalizes to multi-task manipulation)

| Parameter | Value |
|-----------|-------|
| Tasks | 10 (standard MT10 benchmark: reach, push, pick-place, door-open, drawer-open, drawer-close, button-press, peg-insert-side, window-open, window-close) |
| Models | BTWM, LeWorldModel, DreamerV3 |
| Seeds per task × model | 5 |
| Environment steps | 1,000,000 |
| Total runs | 10 × 3 × 5 = **150 runs** |
| Metric | Success rate (%), steps to 80% success |
| Success criterion | BTWM success rate > LeWorldModel by ≥ 5 percentage points |

**Output:** Table 3 (aggregate success rate + steps to 80%).

---

### Block 6 — Analysis: Mutual Information & Bidirectional Consistency [P2]

**Claim:** SC-5 (measurable MI benefits)

**Description:** Post-hoc analysis quantifying information content of latent representations.

**E6.1 — Mutual Information Estimation**
- Train an action classifier on frozen BTWM vs. LeWorldModel latents $[z_t; z_{t+1}]$
- Report classification accuracy as proxy for $I(a_t; z_t, z_{t+1})$
- Hypothesis: BTWM latents preserve more action-relevant information

**E6.2 — Bidirectional Consistency Score**
- Forward: predict $\hat{z}_{t+1}$ from $(z_t, a_t)$
- Inverse: predict $\hat{a}_t$ from $(z_t, z_{t+1})$
- Consistency check: cyclically verify $a_t \rightarrow \hat{z}_{t+1} \rightarrow \hat{a}_t$ chain
- Hypothesis: BTWM's cycle-consistency error is lower than LeWorldModel's

| Parameter | Value |
|-----------|-------|
| Games | 10 (analyze final checkpoints from Block 1) |
| Compute | ~0.1 GPU-day (classifier training only) |
| Runs | 0 new training runs required |

**Output:** Figure 6 (MI bar chart) + inline analysis in §5.5.

---

## 4. Run Count Summary

| Block | Priority | Runs | GPU-Days (A100) | Wall-Time (4× parallel) |
|-------|----------|------|-----------------|------------------------|
| B1: Atari main | P0 | 200 | 2.5 | 0.6 days |
| B2: Ablation | P0 | 300 | 3.8 | 1.0 days |
| B3: Controls | P0 | 100 | 1.3 | 0.3 days |
| B4: DMC | P1 | 135 | 2.0 | 0.5 days |
| B5: Meta-World | P1 | 150 | 8.0 | 2.0 days |
| B6: Analysis | P2 | — | 0.1 | 0.1 days |
| **Total** | | **885** | **17.7** | **~4.5 days** |

---

## 5. Run Order

```
Phase 1 (Critical Path — P0 blocks):
  Day 1:  Launch B1 (Atari main) + B2 (Ablation) in parallel
          → B1 finishes first (~0.6 days); B2 continues
  Day 2:  B2 completes; launch B3 (Controls) immediately
          Analyze B1 results → determine if B4/B5 need adjustment

Phase 2 (Generalization — P1 blocks):
  Day 3:  Launch B4 (DMC) + B5 (Meta-World) in parallel
  Day 5:  B4 completes; B5 continues

Phase 3 (Analysis — P2):
  Day 5:  Launch B6 (MI analysis) using B1 checkpoints
          → completes same day

Contingency:
  - If B1 shows IQM(BTWM) ≤ IQM(LeWorldModel) + 0.03 → stop, investigate
  - If B2 shows inverse loss Δ < 0.05 → all other blocks lower priority; re-examine hypothesis
  - If B3 (compute-matched LeWM) closes >50% of gap → paper claim needs weakening
```

---

## 6. Success Criteria Summary

| Gate | Criterion | Action if Failed |
|------|-----------|-----------------|
| G1 | IQM(BTWM) > IQM(LeWM) + 0.10 | Proceed to B2-B3 |
| G2 | Δ(−Inverse) > 0.10 IQM, and Δ(−Inverse) > all other single ablations | Proceed to B4-B5 |
| G3 | Compute-matched LeWM IQM ≤ LeWM + 0.03 | Claim PC-1 clean |
| G4 | Random-head IQM ≈ LeWM IQM | Claim SC-4 clean |
| G5 | B4+B5: BTWM > LeWM on ≥70% of tasks | Generalization claim SC-6 solid |

---

## 7. Key Design Decisions

1. **10 games over 26:** The 26-game set includes several games where all methods saturate near ceiling (e.g., Assault, Crazy Climber) or floor (e.g., Frostbite). The 10-game selection maximizes statistical power by focusing on games where inverse prediction is *hypothesized* to matter: ambiguous transitions, large action spaces, multi-tasking. A 10-game IQM with 5 seeds provides adequate statistical power (stratified bootstrap with 10 strata).

2. **5 seeds, not 3:** Atari 100k has high variance. 5 seeds is the minimum for reliable IQM with stratified bootstrap CIs. Running only 3 seeds risks inconclusive results.

3. **Compute-matched control is essential:** Without it, reviewers will claim "any extra capacity helps." We must show that 15.1M-param LeWorldModel ≠ LeWorldModel + 0.2M-param inverse head.

4. **Random-head control is the kill shot:** This directly addresses "any auxiliary task helps" — a common reviewer critique for multi-task representation learning papers. If a random-label classifier head gives zero gain, the inverse signal is structurally meaningful.

5. **DMC and Meta-World are P1 (not P0):** The paper's core contribution is the inverse-action mechanism. Generalization to continuous control is supporting evidence, not the primary claim. If compute is tight, these can be reduced to 3 seeds or deferred to supplementary.
