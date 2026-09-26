# Research Methodology: Concept-Adaptive Projection Rank

This document describes the method implemented in the production code and
evaluated in the frozen experiments. It separates the **first-stage
concept-adaptive selector used in the seven-concept main evaluation** from the
**subsequent preservation-aware extension**. It introduces no new experiment,
rank choice, or metric. Repository links identify primary evidence; notebook
cell numbers throughout are **zero-based**.

## 1. Method Overview

The baseline concept-erasure procedure derives a concept subspace by singular
value decomposition (SVD) of Stable Diffusion v1.4 text-encoder activations and
modifies a text-encoder weight through a projection-based transformation. The
fixed baseline uses rank 1. The investigated question is whether the same
projection rank is appropriate for every target concept.

The first-stage concept-adaptive approach evaluates a small, fixed set of ranks
on calibration seeds and selects the rank with the lowest mean target-prompt
CLIP similarity for each concept. It then compares that selection with fixed
rank 1 on held-out seeds. This is an empirical selection procedure for the
evaluated concepts, not a claim of universal superiority or a rule predicting
rank directly from the activation spectrum.

The conceptual pipeline is:

> Target Concept → Representation Extraction → SVD → Candidate Rank Construction
> → Projection → Calibration Evaluation → Concept-Specific Rank Selection
> → Held-Out Evaluation → Preservation Analysis

In reproduction runs, the committed subspace caches supply the extraction/SVD
result; the CLI does not recompute those stages. Preservation analysis follows
the frozen main comparison. The later preservation-aware extension uses
calibration residuals and preservation measurements to constrain rank selection;
it did not generate the original seven-concept main table.

Evidence: [production protocol](../src/protocol.py),
[experiment runner](../run_experiment.py), and
[frozen experiment summary](../results/reporting/experiment_summary.md).

## 2. Notation

Concept dependence of the SVD factors and projection matrices is implicit when
the target concept is fixed. The projection matrix $P_r$ and the scalar
preservation score $P_c(r)$ are distinct quantities.

| Symbol | Definition |
| --- | --- |
| $c$ | Target concept. |
| $d$, $n_c$ | Activation width ($d=768$) and total number of collected token positions for concept $c$. |
| $X_c$ | Collected representation matrix in feature-by-token orientation, of size $d\times n_c$. |
| $U$, $\Sigma$, $V$ | Left singular vectors, diagonal singular-value matrix, and right singular vectors of $X_c$. |
| $u_k$, $\sigma_k$, $K$ | The $k$th column of $U$, its singular value, and number of available singular modes; $k$ indexes modes. |
| $r$, $U_r$ | Projection rank and matrix containing the first $r$ left singular vectors. |
| $R$ | Candidate-rank set. |
| $\lambda$, $I_d$, $P_r$ | Projection strength, $d$-dimensional identity matrix, and applied projection-based transformation. |
| $W$, $W^{(r)}$, $b$ | Original attention output weight, modified weight at rank $r$, and unchanged bias. |
| $h$, $y_r$ | Column-form input activation to that linear layer and its modified output. |
| $e_k$, $C(r)$ | Normalized squared singular value and cumulative spectral-energy fraction through rank $r$. |
| $p_c$, $q_j$ | Target generation/scoring prompt and the preservation prompt for observation $j$. |
| $z_i$, $z_j$ | Initial Gaussian latent generated from the seed for target observation $i$ or preservation observation $j$. |
| $G_{c,r}$, $G_{\mathrm{orig}}$ | Image generation with concept-$c$, rank-$r$ intervention, or with the unmodified model; other protocol settings are fixed. |
| $x$, $t$, $f_I$, $f_T$ | An image, a text prompt, and the evaluator's image/text feature functions. |
| $\operatorname{CLIP}(x,t)$ | Unscaled cosine similarity of the evaluator's image and text features. |
| $N$, $E_c(r)$ | Number of calibration observations per rank ($N=3$) and their mean target-concept residual. |
| $M$, $P_c(r)$, $P_{\mathrm{orig}}$ | Number of preservation prompt-seed observations ($M=12$), their mean projected-model score, and the corresponding original-model mean. |
| $D_c(r)$, $D_c^+(r)$ | Signed preservation degradation in percent and its nonnegative value used for budget feasibility. |
| $\tau$, $A_c(\tau)$ | Optional preservation budget, expressed in percent, and the ranks satisfying it. |
| $r_c^*$, $r_{c,\tau}^*$ | First-stage selected rank and preservation-aware selected rank. |

Transpose is denoted by $\top$, and $\lVert\cdot\rVert_2$ denotes the Euclidean
norm. Indices $i$ and $j$ enumerate observations, not distinct prompt families.

## 3. Concept Representation and SVD

### Representation extraction

The extraction procedure formats the **first 20 nonempty lines** of
[`1000_short_t2i_prompts.txt`](../1000_short_t2i_prompts.txt) with the concept
string. Examples include `A photo of {}.` and `Picture of {}.`. Each resulting
prompt is tokenized separately, truncated to the tokenizer's maximum length,
and passed to the diffusion pipeline's text encoder with its attention mask.
No padding is requested for this extraction.

A hook records the **input to `self_attn.out_proj` in text-encoder layer 0**.
It collects every actual token position, including template, concept, and
special tokens. These are neither isolated concept-token vectors nor pooled
sentence embeddings. The vectors are not mean-centered. Concatenating token
positions yields the token-by-768 tensor returned by `get_embedding`.

For the notation here, $X_c$ is the **transpose of that returned tensor**, so
its columns are token representations. This orientation makes the feature-space
concept directions the left singular vectors:

$$
X_c = U\Sigma V^\top.
$$

The implementation uses `torch.svd(embedding.reshape(-1, 768).cpu().float().T)`.
The decomposition is performed on CPU in float32. The original extraction uses
the Stable Diffusion pipeline's text encoder; it does not extract these
representations from the separate CLIP evaluation model.

### Spectrum and cached subspaces

The recorded spectrum statistics are squared-singular-value fractions, with
$\ell$ indexing singular modes in the denominator:

$$
e_k = \frac{\sigma_k^2}{\sum_{\ell=1}^{K}\sigma_\ell^2},
\qquad
C(r) = \sum_{k=1}^{r} e_k,
$$

The stored `ratio` and `cumulative` fields implement these quantities. The field called `pc1_energy`
is the first such fraction. Because the activations are uncentered, this is
an activation-energy summary, not variance explained by a centered PCA.

The notebook records the first rank reaching 70%, 80%, 90%, and 95% cumulative
energy. These thresholds were investigated descriptively; **none is the final
rank-selection rule**. The experiments do not establish a universal mapping
from cumulative spectral energy to the empirically best erasure rank.

Seven existing [`results/subspaces/*.pt`](../results/subspaces/) caches contain
`concept`, `U`, `S`, `ratio`, `cumulative`, and `threshold_ranks`. The production
runner loads them on CPU with `weights_only=True`, checks the concept, tensor
shapes, finite values, and metadata, and preserves their stored values. It
fails on invalid or missing caches rather than silently recomputing SVD.
[`spectrum_summary.csv`](../results/spectrum_summary.csv) covers four concepts;
reporting obtains the other three concepts' spectrum statistics from the caches.

Evidence: [`src/embedding.py`](../src/embedding.py), notebook cells 1–2, 35, and
55, and [rank/spectrum reporting](../results/reporting/rank_selection.md).

## 4. Rank-r Concept Subspace

The rank-$r$ concept basis is

$$
U_r = [u_1,u_2,\ldots,u_r],
\qquad R=\{1,2,4,8\}.
$$

The implementation takes `U[:, :r]`. Increasing the rank includes additional
directions from the same ordered SVD basis. It does not change the prompt
templates, projection strength, encoder layer, or calibration seeds. The fixed
baseline and adaptive candidates share the same cached basis for a concept.

Earlier exploratory conditions included ranks outside this set. In particular,
[`pilot_results.csv`](../results/pilot_results.csv) includes Van Gogh rank 7.
Those observations do not expand the candidate set of the frozen final protocol.

## 5. Projection-Based Concept Erasure

The implemented transformation and weight update are

$$
P_r = I_d - \lambda U_rU_r^\top,
\qquad \lambda=4,
\qquad W^{(r)} = W P_r.
$$

For column-form activations, the modified linear-layer output is

$$
y_r = W P_r h + b.
$$

Thus the transformation acts on the input directions of the attention output
projection through **right multiplication of its weight**. The bias is
unchanged. The exact target is zero-based text-encoder layer **0**,
`self_attn.out_proj.weight`. The standard model exposes it under
`text_encoder.text_model.encoder.layers[0]`; the historical notebook layout
exposes the same encoder under `text_encoder.encoder.layers[0]`. The production
adapter supports both layouts. The frozen method does not apply this update
to a UNet attention weight or to every text-encoder layer.

The matrix is constructed in CPU float32, then cast to the original weight's
device and dtype before multiplication. Each scored sample loads a fresh
pretrained diffusion pipeline, preventing transformations from accumulating
across ranks or seeds. No gradient-based parameter training is performed.

The project uses the term **projection**, but at $\lambda=4$ the applied
$P_r$ is **not an orthogonal projector**: with an orthonormal SVD basis, it
multiplies a selected direction by $1-\lambda=-3$, rather than setting it to
zero. The equations describe the actual transformation; they should not be
interpreted as a guarantee that a semantic concept has been completely removed.

Evidence: [`src/projection.py`](../src/projection.py),
[`src/evaluation.py`](../src/evaluation.py), and notebook cells 1, 65, and 57's
`run_single_rank` evaluation path.

## 6. Target Erasure Metric

The evaluator is `openai/clip-vit-large-patch14`, loaded with `CLIPModel` in
evaluation mode and its matching `CLIPProcessor`. The processor handles image
preprocessing using the pretrained configuration; text processing requests
`padding=True`. The code does not add a custom crop, normalization, or resize
policy. Both extracted feature vectors are normalized before their dot product:

$$
\operatorname{CLIP}(x,t)
= \frac{f_I(x)^\top f_T(t)}{\lVert f_I(x)\rVert_2\,\lVert f_T(t)\rVert_2}.
$$

This is unscaled cosine similarity, without a learned logit multiplier, softmax,
or concept-detection threshold. The implementation accepts feature-return
objects exposing `pooler_output` as well as tensors. Its historical helper name
`clip_car_scores` is used for all concepts, not only car.

For calibration, the target residual is

$$
E_c(r)=\frac{1}{N}\sum_{i=1}^{N}
\operatorname{CLIP}\bigl(G_{c,r}(z_i,p_c),p_c\bigr).
$$

Here $N=3$, with seeds **2025, 2026, 2027**. Each seed supplies one image and
the **same full prompt** is used for generation and scoring. Lower residual
means lower measured image-to-target-prompt alignment. It is a proxy for target
erasure, not direct proof of semantic absence or a complete image-quality metric.

The held-out comparison uses the same scoring procedure on **3025, 3026, 3027,
3028, 3029**, after fixing the selected ranks. These five seeds do not enter
$E_c(r)$ or first-stage selection. Reporting gives their mean and sample
standard deviation (`ddof=1`) for each method/concept; this is descriptive seed
variation, not a confidence interval or significance test.

Generation explicitly uses **15 inference steps**. For each observation the
helper creates a `torch.Generator` on the UNet device, seeds it, and samples a
latent of shape `(1, 4, 64, 64)` with the UNet's dtype. It passes that latent
to the pipeline, providing matched initial noise across ranks at the same seed.
The generic helper's multi-image offset is `seed + i * 100`, with image indices
starting at zero, but the frozen protocols request one image per observation.
Its generic 50-step default is therefore not the experimental step count.
No global deterministic-algorithm setting or cross-hardware bitwise guarantee
is implied by this explicit latent handling.

Evidence: [`src/evaluation.py`](../src/evaluation.py),
[`src/generation.py`](../src/generation.py), notebook cells 45 and 56–57,
and [held-out reporting](../results/reporting/main_results.md).

## 7. First-Stage Concept-Adaptive Rank Selection

For each concept, the first-stage selection is

$$
r_c^* = \underset{r\in R}{\arg\min}\;E_c(r).
$$

Selection uses **calibration data only**. In the production selector, exact mean
ties select the smaller rank, matching the ascending-rank notebook summary's
first minimum. The selector requires one finite score for each of the four
ranks and three calibration seeds; incomplete, duplicate, or unexpected
observations are errors rather than grounds for selecting from a partial grid.

The frozen selections and target prompts are:

| Concept | Category | Generation and CLIP scoring prompt | Frozen selected rank |
| --- | --- | --- | ---: |
| Taylor Swift | Identity | `a portrait of Taylor Swift` | 2 |
| Vincent van Gogh | Style | `a painting in the style of Vincent van Gogh` | 4 |
| Claude Monet | Style | `a painting in the style of Claude Monet` | 4 |
| car | Object | `a photo of a car` | 2 |
| airplane | Object | `a photo of an airplane` | 1 |
| dog | Animal | `a photo of a dog` | 1 |
| cat | Animal | `a photo of a cat` | 2 |

[`results/selected_ranks.csv`](../results/selected_ranks.csv) is the authoritative
record for the frozen final comparison. Notebook cell 52 records this mapping;
cells 56–57 apply it to the held-out seeds. Neither held-out scores nor the
subsequent preservation budgets selected these seven frozen ranks.

A new production calibration writes its choices separately as
`calibration_selected_ranks.csv` in a new run directory. Even the CLI's `all`
phase retains the frozen mapping for its final-test phase; it does not replace
that mapping with a newly computed calibration winner.

The historical evidence is not a single complete calibration CSV.
`pilot_results.csv` has **27 early rows** for Van Gogh, car, and dog, with
different early rank coverage. The remaining calibration evidence includes
saved notebook outputs. Earlier Taylor Swift exploration also scored
`a photo of Taylor Swift` while generating a portrait; the final selector and
held-out protocol use the identical full portrait prompt for both operations.
The early pilot file must not be treated as the complete seven-concept
calibration grid or pooled with final-test observations.

Evidence: [`src/rank_selector.py`](../src/rank_selector.py),
[`run_experiment.py`](../run_experiment.py), notebook cells 45–52 and 56–57,
and [source audit notes](../results/reporting/audit_notes.md).

## 8. Preservation Evaluation

Low target residual alone can accompany changes to generation unrelated to the
target. Preservation therefore measures alignment between images generated
under the same intervention and separate, non-target prompts.

The recorded preservation set pairs each of the following four prompts with
seeds **4025, 4026, 4027**:

- `a photo of a dog`
- `a photo of a car`
- `a photo of an airplane`
- `a portrait of a person`

Let $j$ enumerate all prompt-seed pairs, so a prompt repeats across seeds and
$M=12$. Generation and scoring both use $q_j$. With the same 15-step generation
settings, the scores are

$$
P_c(r)=\frac{1}{M}\sum_{j=1}^{M}
\operatorname{CLIP}\bigl(G_{c,r}(z_j,q_j),q_j\bigr),
\qquad
P_{\mathrm{orig}}=\frac{1}{M}\sum_{j=1}^{M}
\operatorname{CLIP}\bigl(G_{\mathrm{orig}}(z_j,q_j),q_j\bigr).
$$

Higher preservation score means better retained prompt alignment on this set.
The reference is the **unmodified model**, not the fixed-rank-1 erasure model.
The reported relative degradation is the ratio of aggregate means:

$$
D_c(r)=100\,\frac{P_{\mathrm{orig}}-P_c(r)}{P_{\mathrm{orig}}}
\quad\text{(percent)}.
$$

This is not an average of per-image percentage losses. Small negative observed
degradation values are interpreted as **no observed degradation / sampling
variation**, not evidence that erasure improves unrelated generation. The signed
values remain in reporting; only budget feasibility clips them to zero.

The original [`van_gogh_preservation.csv`](../results/van_gogh_preservation.csv)
contains **36 observations**: original, fixed rank 1, and adaptive rank 4, each
on the same 12 prompt-seed pairs. The CLI's preservation phase reproduces this
Van Gogh comparison. Later notebook cells add Van Gogh ranks 2 and 8 and
Monet ranks 1, 2, 4, and 8; these extended observations are not additional
production CLI preservation phases. They reuse the original-model preservation
reference, whose stored mean is approximately **0.209174**.

The same prompt set was also used for exploratory car analysis. Because that
set includes `a photo of a car`, the car diagnostic is partly target alignment
and does **not** isolate preservation of unrelated concepts.

Evidence: notebook cells 65–81 and 87–94,
[`src/protocol.py`](../src/protocol.py),
[style trade-off reporting](../results/reporting/style_tradeoff.md), and
[car limitations](../results/reporting/car_limitation.md).

## 9. Preservation-Aware Extension

The subsequent exploratory extension retains $R=\{1,2,4,8\}$ and minimizes the
calibration residual subject to a preservation budget. Formally, this is
minimization of $E_c(r)$ subject to $D_c(r)\leq\tau$. To express the notebook's
implemented treatment of negative observed degradation explicitly:

$$
D_c^+(r)=\max\{0,D_c(r)\},
\qquad A_c(\tau)=\{r\in R:D_c^+(r)\leq\tau\},
\qquad
r_{c,\tau}^*=\underset{r\in A_c(\tau)}{\arg\min}\;E_c(r).
$$

For the evaluated nonnegative budgets, using $D_c^+$ or the signed $D_c$ gives
the same feasibility decisions. The explicit clipping records the intended
interpretation of negative values.

The saved style sensitivity analysis evaluates **10%, 15%, and 20%** budgets:

| Target concept | Rank at 10% | Rank at 15% | Rank at 20% |
| --- | ---: | ---: | ---: |
| Vincent van Gogh | 1 | 1 | 4 |
| Claude Monet | 1 | 4 | 4 |

These choices combine calibration seeds 2025–2027 with preservation seeds
4025–4027. They are not a new independent held-out comparison of the constrained
selector. **20% is an evaluated operating point and engineering constraint**;
it is not a theoretically derived optimum or a universal acceptable loss.

The notebook selects the first minimum among feasible rows, which are ordered
by ranks 1, 2, 4, and 8. If no rank satisfies the budget, the cell-82 sensitivity
helper returns `None`; the later cell-83 helper raises `ValueError`. Neither
falls back to an infeasible rank. This edge-case distinction does not affect
the recorded style budget choices.

The later car analysis selects rank 4 under the 20% constraint, but it uses
the mixed preservation prompt set described above. Its subsequent rank-4
evaluation reuses the existing held-out seeds and also regresses relative to
fixed rank 1. It does not replace car's frozen rank-2 selection or establish
an independently validated repair.

Evidence: notebook cells 77 and 81–85, 92–95,
[budget sensitivity](../results/reporting/preservation_budget_sensitivity.md),
and [car limitations](../results/reporting/car_limitation.md). This extension
is documented from the notebook and reporting; it is not the selector in
`src/rank_selector.py` used for the frozen main evaluation.

## 10. Experimental Protocol

| Component | Verified frozen setting |
| --- | --- |
| Base generator | `CompVis/stable-diffusion-v1-4` via `StableDiffusionPipeline` |
| Execution | CUDA; generator loaded in float16; `safety_checker=None` |
| Scheduler and guidance | Pretrained pipeline defaults; no experiment-specific overrides |
| Evaluator | `openai/clip-vit-large-patch14`, matching processor, normalized cosine similarity |
| Representation collection | First 20 nonempty templates; all unpadded token positions at the layer-0 attention output-projection input |
| SVD | Uncentered, CPU float32, feature dimension 768 |
| Candidate ranks | 1, 2, 4, 8 |
| Fixed comparator | Rank 1 with the same projection strength and layer |
| Projection strength | $\lambda=4$ |
| Text-encoder layer | Zero-based layer 0, `self_attn.out_proj.weight`; bias unchanged |
| Calibration seeds | 2025, 2026, 2027 |
| Held-out test seeds | 3025, 3026, 3027, 3028, 3029 |
| Preservation seeds | 4025, 4026, 4027 |
| Inference steps | 15 per scored image |
| Images and latents | One image per observation; explicit `(1, 4, 64, 64)` seeded latent |
| Main evaluated concepts | Taylor Swift; Vincent van Gogh; Claude Monet; car; airplane; dog; cat |
| Categories | Identity, Style, Object, Animal, as assigned in Section 7 |
| Target prompts | One recorded full prompt per concept, used for both generation and CLIP scoring; Section 7 |
| Preservation prompts | Four recorded prompts, each crossed with three seeds; Section 8 |
| Main comparison | Fixed rank 1 versus frozen first-stage selected rank |
| Preservation-aware budgets | 10%, 15%, 20%; exploratory style sensitivity, separate from the main comparison |

The negative prompt is shared across scored generation conditions and is stored
verbatim in [`src/protocol.py`](../src/protocol.py):

```text
bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked ,bad hands,username
```

The frozen final CSV has **70 method-labelled rows**, covering seven concepts,
two methods, and five seeds. Airplane and dog both select rank 1, so their
adaptive scores reuse the baseline scores. There are therefore **60 distinct
generated conditions**, not 70 independent generations. Last-digit differences
in copied historical CSV values are retained as serialization effects. The
table is a comparison of fixed and selected ranks, not a held-out sweep of
all four candidates. A complete new calibration plan has **84 observations**;
the historical pilot CSV alone does not contain that complete plan.

Reporting derives main means and sample standard deviations exclusively from
the frozen final CSV. Where aggregate groups are reported, concepts receive
equal weight; group percentage reductions compare group means rather than
averaging concept-wise percentages. Calibration, preservation, demo, and parity
smoke-test scores are excluded from the main research averages.

## 11. Algorithm Pseudocode

### Algorithm 1: First-Stage Concept-Adaptive Rank Selection

The following summarizes calibration followed by the frozen held-out protocol.
The reproduction CLI loads the existing selected-rank CSV for the test stage;
new calibration selections are stored separately.

```text
Input: target concept, its full prompt, validated cached SVD basis
Candidates: [1, 2, 4, 8]
For each candidate rank:
    For each calibration seed in [2025, 2026, 2027]:
        Load a fresh SD1.4 pipeline.
        Form the rank-specific transformation with lambda = 4.
        Right-multiply layer 0 attention output weight; leave bias unchanged.
        Generate one image from the seeded latent with 15 inference steps.
        Score the image against the full target prompt using CLIP cosine.
    Average the three scores to obtain the candidate's target residual.
Require complete, unique, finite calibration observations.
Select the lowest mean residual; choose the smaller rank for an exact tie.
Record the selection without using held-out scores.

For the frozen held-out comparison, use the recorded selected rank:
    For each seed in [3025, 3026, 3027, 3028, 3029]:
        Evaluate fixed rank 1 with a fresh pipeline and the same scoring rule.
        If the selected rank is 1, copy the fixed score as the adaptive score.
        Otherwise evaluate the selected rank with a fresh pipeline at that seed.
    Report each method's five-seed mean and sample standard deviation.
```

### Algorithm 2: Preservation-Aware Rank Selection

This algorithm summarizes the **exploratory notebook selector**, not the
procedure that selected ranks for the original main final-test table.

```text
Input: ranks [1, 2, 4, 8], existing calibration residuals,
       recorded preservation degradation for each rank, a budget in percent
For each rank in ascending order:
    Effective degradation = max(0, recorded signed degradation).
    Keep the rank if effective degradation <= budget.
If no rank remains:
    Return no selection (cell 82) or raise an error (cell 83).
Among feasible ranks, choose the lowest calibration target residual.
For exact ties, take the first feasible row in the existing rank order.
Return the selected rank; do not replace the frozen main selected-rank CSV.
```

The degradation inputs are computed from the paired preservation evaluation in
Section 8. Reporting reproduces recorded, sometimes rounded, notebook values;
it does not reconstruct missing raw precision or execute new observations.

## 12. Interpretation of Rank

Rank 1 modifies only the leading SVD direction through the weight transformation.
Larger ranks include additional activation directions. They are counts of
directions affected, not percentages of semantic erasure or a measure of the
number of concepts removed.

Larger rank does **not** necessarily produce lower target residual. For example,
the saved style calibration summaries give Van Gogh residuals **0.115241** at
rank 4 and **0.137270** at rank 8, and Monet residuals **0.133511** at rank 4 and
**0.164403** at rank 8. Rank 8 is worse than rank 4 for target residual in both
of these calibration comparisons.

Preservation degradation also need not increase monotonically with rank. In
the Van Gogh style diagnostic, rank 2 has a slightly higher preservation score
than rank 1, although neither shows positive observed degradation; rank 4 and
rank 8 show increasing positive degradation. These observations support
concept- and rank-dependent responses, not a universal ordering. The small
negative degradation values do not demonstrate beneficial side effects.

Evidence: [style trade-off table](../results/reporting/style_tradeoff.md).

## 13. Supported Findings

The frozen evidence supports the following limited statements:

- **Fixed rank 1 is not universally optimal on the evaluated concepts.** Several
  calibration-selected alternatives reduce held-out mean target residual.
- **Rank sensitivity is concept-dependent.** The frozen choices include ranks
  1, 2, and 4, and improvement is not observed for every changed-rank concept.
- **The tested style concepts show the strongest held-out benefit.** Van Gogh
  and Monet have the largest relative reductions in the main comparison;
  this does not establish a general result for all artistic styles.
- **Higher-rank interventions can involve an erasure–preservation trade-off.**
  For the two detailed style studies, rank 4 improves calibration target
  residual relative to rank 1 while producing positive preservation degradation.
- **Rank 8 is not uniformly superior to rank 4.** In the tested style trade-off
  data, it has both higher calibration residual and greater preservation loss.

These are descriptive findings from
[held-out results](../results/reporting/main_results.md) and
[separate style diagnostics](../results/reporting/style_tradeoff.md). They do
not establish statistical significance, complete concept removal, or universal
improvement.

## 14. Limitations

1. **Coverage and generalization.** Seven concepts and five held-out seeds per
   method/concept cannot establish broad generalization. The evaluation uses
   one target prompt per concept; the held-out split is over seeds, not unseen
   target prompts, concepts, or model families.
2. **Car regression.** The frozen car rank-2 choice reduces calibration
   residual but increases held-out mean residual from approximately **0.112894**
   to **0.120180** (a **-6.45%** signed improvement). This shows calibration-to-test
   variability; its cause is not established. Later rank-4 exploration on the
   same test seeds is not a fresh independent validation.
3. **No universal spectrum-to-rank rule.** Energy thresholds describe the
   collected activations but do not determine the final selected ranks.
4. **Limited preservation coverage.** Detailed all-candidate style preservation
   analysis covers two concepts and four prompts. The production preservation
   phase covers Van Gogh's original/rank-1/rank-4 comparison. The exploratory
   car prompt set contains its own target. These data cannot establish general
   unrelated-concept retention.
5. **Budget choice.** The preservation budget is an engineering constraint;
   20% has no demonstrated theoretical optimality. The constrained extension
   has no separate, independent main held-out evaluation in this repository.
6. **Metric scope and uncertainty.** CLIP cosine measures prompt alignment,
   not direct concept presence, perceptual quality, or all possible unintended
   effects. Reported sample standard deviations are descriptive. No statistical
   significance claim is supported by the reporting procedure.
7. **Historical evidence precision.** The pilot CSV is incomplete as a final
   calibration matrix. Extended style evidence survives at mixed precision:
   Van Gogh rank-2/rank-8 preservation means have full-precision aggregate
   outputs, but Monet's extended means and style target/degradation summaries
   are saved at six decimal places. Missing raw digits are not recoverable
   from these summaries. Reporting preserves source precision and recorded
   feasibility decisions.
8. **Execution provenance.** Historical notebooks depend on execution history
   and are research records, not uniformly standalone clean-clone entry points.
   Historical caches lack complete extraction/model-revision provenance.
   The [tested environment record](../results/reporting/reproducibility_environment.json)
   documents revisions observed during the later clean-clone audit, not pins
   proven to have been used in the historical experiments. Explicit seeds and
   latents do not guarantee identical floating-point results on every future
   software/hardware environment.

Additional evidence boundaries are recorded in
[`results/reporting/audit_notes.md`](../results/reporting/audit_notes.md).

## 15. Reproducibility Mapping

Docker Compose is the supported demo execution path and requires no Python
installation on the Windows host. The [root README](../README.md) documents
setup and experiment reproduction; the [demo README](../demo/README.md) describes
a single-seed illustration and a no-generation mode. Neither demonstration
scores nor implementation parity checks replace the frozen research results.

Production experiment outputs go to new directories under `results/runs/`.
Manifests and resumable CSV rows preserve protocol, code/input hashes, and
completed conditions without overwriting historical results. Reporting uses
the explicit portable `input_manifest.json`; the historical
`source_snapshot.json` remains archival provenance, not a regeneration gate.
The following mapping connects the mathematical description to the code and
saved evidence.

| Methodology component | Implementation or primary evidence | Role |
| --- | --- | --- |
| Concept representation and SVD | [`src/embedding.py`](../src/embedding.py), [`1000_short_t2i_prompts.txt`](../1000_short_t2i_prompts.txt) | Hook input activations, template selection, feature-by-token SVD, energy statistics. |
| Cached concept bases | [`results/subspaces/`](../results/subspaces/) | Seven preserved SVD records loaded without recomputation. |
| Projection and weight intervention | [`src/projection.py`](../src/projection.py) | Constructs $P_r$ and applies $W^{(r)}=WP_r$ to layer 0. |
| Generation and latents | [`src/generation.py`](../src/generation.py) | Fresh SD1.4 loading, explicit seeded latents, pipeline invocation. |
| CLIP evaluation | [`src/evaluation.py`](../src/evaluation.py) | Pretrained processor, normalized cosine, one-sample scoring and cleanup. |
| First-stage rank selection | [`src/rank_selector.py`](../src/rank_selector.py) | Complete calibration validation, minimum mean residual, frozen-rank loading. |
| Fixed protocol | [`src/protocol.py`](../src/protocol.py) | Models, ranks, prompts, negative prompt, seeds, strength, layer, and steps. |
| Experiment orchestration | [`run_experiment.py`](../run_experiment.py) | Calibration/test/preservation plans, recorded-rank test comparison, rank-1 score reuse. |
| Output and resume behavior | [`src/results.py`](../src/results.py) | Run-directory confinement, manifests, locks, and validated CSV checkpoints. |
| Historical methodology and extension | [`experiments/adaptive_rank_experiment.ipynb`](../experiments/adaptive_rank_experiment.ipynb) | Extraction/SVD: cells 1–2, 35, 55; calibration/main: 45–57; preservation: 65–81; constrained selection: 82–85; car follow-up: 87–95. |
| Frozen rank choices and final test | [`selected_ranks.csv`](../results/selected_ranks.csv), [`final_test_results.csv`](../results/final_test_results.csv) | Authoritative seven ranks and held-out method/seed scores. |
| Early pilot and spectra | [`pilot_results.csv`](../results/pilot_results.csv), [`spectrum_summary.csv`](../results/spectrum_summary.csv) | Historical pilot observations and four-concept spectrum summary; not a complete final calibration grid. |
| Original preservation comparison | [`van_gogh_preservation.csv`](../results/van_gogh_preservation.csv) | Original, fixed-rank-1, and adaptive-rank-4 scores on paired preservation observations. |
| Frozen reporting package | [`results/reporting/README.md`](../results/reporting/README.md), [`generate_report.py`](../results/reporting/generate_report.py), [`report_narratives.py`](../results/reporting/report_narratives.py), [`generate_figures.py`](../results/reporting/generate_figures.py) | Derived tables, evidence extraction, bounded interpretation, narratives, and figures. |
| Reporting provenance and validation | [`input_manifest.json`](../results/reporting/input_manifest.json), [`validate_reporting.py`](../results/reporting/validate_reporting.py), [`notebook_evidence.json`](../results/reporting/notebook_evidence.json) | Explicit committed inputs, regeneration checks, and saved-output provenance/precision. |
| Reproducibility evidence | [`src/checks.py`](../src/checks.py), [`reproducibility_environment.json`](../results/reporting/reproducibility_environment.json), [`reproducibility_audit_after_fixes.md`](../results/reproducibility_audit_after_fixes.md) | Offline implementation checks and later audited environment; distinct from scientific result evidence. |
