# MCSL: Multi-Modal Contrastive Sequential Learning for Recommendation

A 3-tier sequential recommendation system built on the Amazon Reviews 2023 (Video Games) dataset, progressing from a standard Transformer-based recommender (SASRec) to a dual-encoder model that fuses learnable item-ID embeddings with frozen LLM semantic embeddings via a contrastive alignment objective.

## Why this project

Most SASRec tutorials/implementations stop at reporting **sampled-ranking metrics** (1 positive item vs. a handful of random negatives). This project shows why that's a problem: under sampled ranking, a model that discards item identity entirely (Tier 2) looks like the second-best model. Under full-catalog ranking (~25.7k items), it's actually the **worst** of the four models tested. This is a known but under-discussed issue in recommender systems evaluation (Krichene & Rendle, *"On Sampled Metrics for Item Recommendation,"* KDD 2020) — sampled metrics don't just add noise, they can reorder models. This project reports full-ranking metrics as primary throughout, and includes the sampled numbers only for reference/comparability with prior work.

## Dataset

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab), Video Games category.

| | Raw | After 5-core filtering |
|---|---|---|
| Interactions | 4,570,961 | 828,932 |
| Users | 2,766,656 | 95,899 |
| Items | 137,249 | 25,708 |

Iterative k-core filtering (5-core on both users and items, looped to a fixed point) and leave-one-out train/val/test splits per user, following the standard SASRec protocol.

## Model tiers

| Tier | Model | Description |
|---|---|---|
| 1 | SASRec | Standard Transformer-based sequential recommender (self-attentive, causal masking) |
| 1 | SASRec+Time | Adds log-scale bucketed time-gap embeddings between consecutive interactions |
| 2 | LLM-SASRec | Replaces item-ID embeddings with frozen `all-MiniLM-L6-v2` (384-dim) semantic embeddings of item text (title, store, features) |
| 3 | **MCSL** | Dual encoder: sums learnable ID embeddings + time embeddings + projected semantic embeddings, with an added InfoNCE contrastive loss aligning the sequence context vector to the target item's semantic embedding |

## Evaluation protocol

Two ranking protocols were used; **full-ranking is treated as the primary, reported metric**:

- **Sampled ranking**: 1 positive vs. 99 negatives, excluding items already in the user's history (matches training-time negative sampling).
- **Full ranking** (primary): rank the true next item against the entire ~25,708-item catalog, evaluated on 5,000 held-out users.

All results are averaged over 3 random seeds (42, 123, 7), with early stopping on validation NDCG@10 (patience 5) and checkpointing of the best-validation epoch.

## Results

**Full-ranking Test metrics, mean ± std across 3 seeds:**

| Model | HR@10 | NDCG@10 |
|---|---|---|
| **MCSL (Tier 3)** | **3.33% ± 0.22pp** | **0.0176 ± 0.0006** |
| SASRec+Time (Tier 1) | 2.31% ± 0.07pp | 0.0117 ± 0.0008 |
| SASRec (Tier 1) | 2.11% ± 0.22pp | 0.0105 ± 0.0010 |
| LLM-SASRec (Tier 2) | 1.73% ± 0.08pp | 0.0082 ± 0.0004 |
| Popularity baseline | 2.28% | 0.0117 |

MCSL improves full-ranking NDCG@10 by **+66.7% relative to plain SASRec** and **+49.9% relative to a popularity baseline** — notable given that plain SASRec itself barely beats the popularity baseline on this metric, a well-documented pattern in the RecSys literature (popularity is a notoriously strong baseline).

A paired t-test across the 3 seeds (full-ranking NDCG@10, MCSL vs. each other model) supports the gap: p = 0.014 vs. SASRec, p = 0.029 vs. SASRec+Time, p = 0.0017 vs. LLM-SASRec. (n=3 seeds — read as a supportive signal, not a strong statistical claim.)

Under the easier *sampled*-ranking protocol, all four models look much closer together and LLM-SASRec even outranks both Tier-1 models — this divergence between the two protocols is itself one of this project's findings, not just a footnote.

## Ablation study (isolating MCSL's components)

MCSL differs from LLM-SASRec in three ways simultaneously (adds ID embeddings, time embeddings, and a contrastive loss), so a component-removal ablation (single seed, full-ranking NDCG@10) was run to see which piece actually matters:

| Variant | Full NDCG@10 | Change |
|---|---|---|
| MCSL (full) | 0.0155 | — |
| w/o time embedding | 0.0157 | ~no effect |
| w/o contrastive loss | 0.0137 | −0.0018 |
| w/o semantic fusion | 0.0105 | **−0.0050 (collapses to ~SASRec level)** |

**Finding: semantic fusion (combining ID and text embeddings in the input) is the dominant driver of MCSL's improvement, not the contrastive loss.** The contrastive objective adds a real secondary boost on top of that; the time embedding contributes negligibly on this metric. This ablation was run on a single seed to keep runtime reasonable — a full multi-seed ablation would be the natural next step to firm this up further.

## Qualitative results

Aggregate metrics don't show *why* one model beats another — a sample of 5 held-out users makes it concrete. SASRec's top-5 recommendations are **nearly identical across all 5 users regardless of their purchase history** (e.g. "Halo 5: Guardians," "Nintendo Switch Pro Controller," "PS4 500GB Console" recur for almost every user) — a sign of popularity collapse rather than personalization. MCSL's recommendations visibly track each user's context (e.g. a user who bought gaming mice and a racing wheel gets recommended more peripherals; a user with an RPG-heavy history gets RPG-adjacent titles). See the notebook's qualitative-examples section for the full output.

## Limitations (stated honestly)

- MCSL had not fully converged at the 40-epoch cap in any of the 3 seeds (val NDCG was still climbing, though slowly, at the final epoch) — the reported numbers likely understate MCSL's true ceiling.
- The ablation study was run on a single seed rather than all 3, for runtime reasons.
- Full-ranking evaluation is capped at 5,000 users (not the full ~96k test set) for tractability; standard error on the reported metrics is roughly ±0.2-0.4 percentage points.
- No hyperparameter search was performed (temperature, lambda_cl, hidden size are fixed at reasonable defaults, not tuned).

## How to reproduce

1. Open the notebook in Google Colab with a GPU runtime (T4 or better).
2. Run all cells top to bottom. Full run (data download, embedding extraction, 3-seed training of 4 models + ablation) takes several hours on a T4.
3. Results are saved to `results.csv` and `ablation_results.csv`; plots to `training_curves.png` and `full_rank_comparison.png`.

## Stack

PyTorch, Hugging Face `datasets` + `sentence-transformers` (MiniLM-L6-v2), pandas, scipy (significance testing).
