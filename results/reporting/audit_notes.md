# Source audit and limitations

The frozen CSVs remain the primary numerical evidence. All source files,
including the existing notebooks and previous smoke-test artifacts, were
hashed before reporting and are checked again after the build.

1. **Main data are complete for the recorded comparison.** The final CSV has
   70 unique concept-method-seed rows: seven concepts, two methods, five test
   seeds. It contains ranks 1, 2, and 4 because adaptive selections are frozen.
   It is not a held-out sweep of all four candidate ranks. The main comparison
   contains 60 distinct generated conditions because airplane and dog reuse
   rank-1 scores for the adaptive method. Tiny last-digit differences in those
   copied CSV values are serialization effects; the original rows remain intact.

2. **Calibration CSV coverage is incomplete.** `pilot_results.csv` has 27 early
   observations: Van Gogh ranks 1/4/7 and car/dog ranks 1/2/4, all on seeds
   2025–2027. It is not the complete seven-concept `{1,2,4,8}` calibration grid.
   Van Gogh rank 7 is an exploratory pilot condition. The main tables do not
   combine these rows with the held-out test.

3. **Extended style evidence survives at mixed precision.** Van Gogh original,
   rank 1, and rank 4 have full-precision sample rows in the preservation CSV.
   Full-precision aggregate means for ranks 2 and 8 survive in notebook cell 76,
   while their individual printed observations have six decimals. Monet's
   extended observations and mean summaries survive only at six decimals in
   cells 79–81. Style calibration means also have six-decimal saved precision.
   No missing raw digits are reconstructed. Recorded degradation percentages
   are used directly; recomputing them from rounded Monet means differs by
   at most about 0.00022 percentage points, a precision issue that does not
   alter any recorded 10/15/20% feasibility decision.

4. **Trade-off target residuals are calibration results.** They come from
   cells 50–51 and match the later constants in cells 77/81/82. Preservation
   uses four prompts with seeds 4025–4027. The combined constraint analysis is
   not an independent held-out comparison of a new rank-selection procedure.
   Negative degradation is reported as no observed degradation / sampling
   variation. It does not establish improved unrelated generation.

5. **Car failure remains visible.** Its original rank-2 selection regresses on
   the held-out seeds. The later 20% budget selecting rank 4 is exploratory and
   does not replace rank 2 in the main tables. Its preservation set includes
   the target Car prompt, so it does not isolate unrelated-concept preservation.
   The subsequent rank-4 evaluation reuses the same held-out seeds and also
   regresses. Its exact aggregate mean and SD survive in cell 95; sample scores
   are printed only to six decimals. Rank-8 calibration values in cell 92 are
   explicitly recovered six-decimal values, not a full-precision CSV source.

6. **Notebook execution history is not a reporting program.** For example,
   cell 89 has a saved `car_full` NameError and is followed by recovery cells.
   Reporting parses saved source/output as data and executes no cells. Full
   dependency/model revision provenance is absent from the historical results.

7. **Spectra are descriptive.** The spectrum CSV covers four concepts; cached
   metadata supplies Monet, airplane, and cat. The overlapping CSV/cache
   statistics agree. No decomposition is recomputed, and no universal rule
   mapping cumulative spectral energy to the best rank is inferred.

8. **Scope is limited.** Seven concepts, five held-out seeds, two detailed
   style preservation studies, and four preservation prompts do not establish
   broad generalization. CLIP cosine is a proxy for target residual and prompt
   alignment, not a complete concept-presence or image-quality assessment.
   Sample SD is descriptive; no significance test, confidence interval,
   unrecorded seed, or new metric is introduced.

No material disagreement was found between the main CSV-derived aggregates
and the saved notebook summaries. Every data block used or excluded is listed
in [data_inventory.csv](data_inventory.csv).
